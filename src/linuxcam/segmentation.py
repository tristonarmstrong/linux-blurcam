"""ONNX-based selfie segmentation for ARM64 Linux."""

import time

import cv2
import numpy as np
import onnxruntime as ort


class FrameTimer:
    """Simple rolling-window timer for per-frame profiling."""

    def __init__(self, window: int = 60):
        self._window = window
        self._samples: dict[str, list[float]] = {}

    def mark(self, label: str, t0: float):
        """Record elapsed time since *t0* under *label*."""
        self._samples.setdefault(label, []).append(time.perf_counter() - t0)
        if len(self._samples[label]) > self._window:
            self._samples[label].pop(0)

    def report(self, prefix: str = "") -> str:
        """Return a formatted string of average timings."""
        parts = []
        for label, vals in sorted(self._samples.items()):
            if not vals:
                continue
            avg = sum(vals) / len(vals) * 1000.0
            parts.append(f"{label}={avg:.1f}ms")
        return f"{prefix} {' | '.join(parts)}" if prefix else " | ".join(parts)

    def reset(self):
        self._samples.clear()


def _now() -> float:
    return time.perf_counter()


class SelfieSegmentation:
    """Selfie segmentation using ONNX Runtime — auto-detects model type."""

    # SINet-specific normalization constants (BGR order, from reference code)
    SINET_MEAN = np.array([102.890434, 111.25247, 126.91212], dtype=np.float32)
    SINET_STD = np.array([62.93292, 62.82138, 66.355705], dtype=np.float32)

    def __init__(self, model_path: str):
        """Initialize ONNX Runtime session and auto-detect model architecture."""
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path, sess_options, providers=providers
        )

        active_providers = self.session.get_providers()
        if active_providers:
            p = active_providers[0]
            print(f"ONNX Runtime using {'GPU' if 'CUDA' in p else 'CPU'}: {p}")

        # Auto-detect model type from input/output shapes
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name

        # Infer spatial dimensions (handle symbolic dims like 'batch_size')
        def _int(dim):
            return dim if isinstance(dim, int) else None

        self.input_shape = [_int(d) for d in inp.shape]
        self.output_shape = [_int(d) for d in out.shape]
        self.input_height = self.input_shape[2] or 256
        self.input_width = self.input_shape[3] or 256

        # Detect model type by output channels
        out_channels = self.output_shape[1]
        if out_channels == 2:
            self.model_type = "sinet"
            print(f"Detected model: SINet ({self.input_height}x{self.input_width})")
        else:
            self.model_type = "mediapipe"
            print(f"Detected model: MediaPipe ({self.input_height}x{self.input_width})")

    def preprocess(self, frame: np.ndarray, timer: FrameTimer | None = None) -> np.ndarray:
        """Preprocess frame for the detected model type."""
        t0 = _now()
        resized = cv2.resize(frame, (self.input_width, self.input_height))

        if self.model_type == "sinet":
            # SINet expects BGR, (img - mean) / std / 255
            img = resized.astype(np.float32)
            img = (img - self.SINET_MEAN) / self.SINET_STD / 255.0
            nchw = np.transpose(img, (2, 0, 1))
        else:
            # MediaPipe expects RGB, /255
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            nchw = np.transpose(normalized, (2, 0, 1))

        batched = np.expand_dims(nchw, axis=0)
        if timer:
            timer.mark("preprocess", t0)
        return batched

    def predict(self, frame: np.ndarray, timer: FrameTimer | None = None) -> np.ndarray:
        """Run inference and return a single-channel confidence mask [0,1]."""
        t0 = _now()
        input_data = self.preprocess(frame, timer=timer)
        t1 = _now()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        if timer:
            timer.mark("inference", t1)
        t2 = _now()

        out = outputs[0][0]  # Remove batch dim

        if self.model_type == "sinet":
            # SINet outputs 2-class logits: [2, H, W]
            # Convert to softmax probabilities, then take foreground channel
            exp = np.exp(out - np.max(out, axis=0, keepdims=True))
            mask = exp[1] / np.sum(exp, axis=0)  # foreground confidence
        else:
            # MediaPipe outputs 1-channel alpha: [1, H, W] or [H, W]
            if len(out.shape) == 3:
                mask = out[0]
            else:
                mask = out

        if timer:
            timer.mark("post-inference", t2)
        return mask

    def get_mask(self, frame: np.ndarray, threshold: float = 0.5, timer: FrameTimer | None = None) -> np.ndarray:
        """Get binary mask resized to original frame size."""
        t0 = _now()
        mask = self.predict(frame, timer=timer)
        t1 = _now()
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        t2 = _now()
        binary_mask = (mask_resized > threshold).astype(np.float32)
        t3 = _now()
        binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
        if timer:
            timer.mark("predict", t0)
            timer.mark("mask-resize", t1)
            timer.mark("mask-threshold", t2)
            timer.mark("mask-blur", t3)
        return binary_mask


def apply_background_blur(
    frame: np.ndarray, mask: np.ndarray, blur_strength: int = 21,
    timer: FrameTimer | None = None,
) -> np.ndarray:
    """Apply blur to background while keeping foreground sharp.

    Background is blurred at 1/3 resolution for speed, then upscaled.
    The composite (blend) runs at full resolution on the GPU via OpenCL
    so the foreground subject stays perfectly sharp.
    """
    t0 = _now()
    if blur_strength % 2 == 0:
        blur_strength += 1

    orig_h, orig_w = frame.shape[:2]

    # Processing resolution: 1/3 scale with minimums
    proc_w = max(320, orig_w // 3)
    proc_h = max(180, orig_h // 3)
    proc_w += proc_w % 2
    proc_h += proc_h % 2

    # Scale kernel to processing resolution so visual strength stays consistent
    kernel_proc = max(3, blur_strength // 3)
    if kernel_proc % 2 == 0:
        kernel_proc += 1

    # 1. Blur background at low resolution
    if orig_w != proc_w or orig_h != proc_h:
        frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        frame_proc = frame

    blurred_proc = cv2.GaussianBlur(frame_proc, (kernel_proc, kernel_proc), 0)

    # 2. Upscale blurred background to full resolution
    if orig_w != proc_w or orig_h != proc_h:
        blurred = cv2.resize(blurred_proc, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    else:
        blurred = blurred_proc

    # 3. Composite at full resolution on GPU (OpenCL)
    #    Upload inputs once, all math stays on the device, download final uint8.
    frame_u = cv2.UMat(frame)
    blurred_u = cv2.UMat(blurred)
    mask_u = cv2.UMat(mask)

    # Expand mask to 3 channels so multiply works with 3-channel images
    mask_3ch_u = cv2.merge([mask_u, mask_u, mask_u])

    # foreground = frame * mask
    fg_u = cv2.multiply(frame_u, mask_3ch_u, dtype=cv2.CV_32F)
    # background = blurred * (1 - mask)
    inv_mask_u = cv2.subtract(1.0, mask_3ch_u)
    bg_u = cv2.multiply(blurred_u, inv_mask_u, dtype=cv2.CV_32F)
    # result = fg + bg
    result_u = cv2.add(fg_u, bg_u)

    # Download from GPU and convert to uint8 numpy array
    result_u8 = cv2.convertScaleAbs(result_u, alpha=1.0, beta=0)
    result = cv2.UMat.get(result_u8)

    if timer:
        timer.mark("blur", t0)
    return result


# ---------------------------------------------------------------------------
# Debug / preview rendering helpers
# ---------------------------------------------------------------------------

_DEBUG_MODES = ("blur", "mask", "edges", "split", "heatmap", "original")


def _overlay_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS in the top-left corner."""
    out = frame.copy()
    label = f"FPS: {fps:.1f}"
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _render_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the binary mask as a 3-channel grayscale image."""
    h, w = frame.shape[:2]
    mask_8u = (mask * 255).astype(np.uint8)
    # Resize to match frame if needed
    if mask_8u.shape[:2] != (h, w):
        mask_8u = cv2.resize(mask_8u, (w, h))
    return cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2BGR)


def _render_edges(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show the original frame with the segmentation contour drawn in green."""
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask_r = cv2.resize(mask, (w, h))
    else:
        mask_r = mask
    mask_8u = (mask_r * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def _render_split(frame: np.ndarray, mask: np.ndarray, blur_strength: int = 21) -> np.ndarray:
    """Left half = original, right half = blurred result."""
    blurred = apply_background_blur(frame, mask, blur_strength)
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)
    out[:, : w // 2] = frame[:, : w // 2]
    out[:, w // 2 :] = blurred[:, w // 2 :]
    # Vertical line divider
    cv2.line(out, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)
    return out


def _render_heatmap(frame: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    """Show the raw segmentation confidence as a color heatmap."""
    h, w = frame.shape[:2]
    if raw_mask.shape[:2] != (h, w):
        raw_mask = cv2.resize(raw_mask, (w, h))
    # Normalize to 0-255
    heat = (np.clip(raw_mask, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat_color


def render_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    raw_mask: np.ndarray,
    mode: str = "blur",
    blur_strength: int = 21,
    fps: float | None = None,
    timer: FrameTimer | None = None,
) -> np.ndarray:
    """Render a frame in the requested debug/preview mode.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame from the webcam.
    mask : np.ndarray
        Binary mask (0-1 float) after threshold + blur, same size as ``frame``.
    raw_mask : np.ndarray
        Raw confidence map from the model before thresholding, may be smaller.
    mode : str
        One of ``blur``, ``mask``, ``edges``, ``split``, ``heatmap``, ``original``.
    blur_strength : int
        Gaussian blur kernel size (odd number).
    fps : float | None
        If given, overlay FPS text on the output.
    timer : FrameTimer | None
        If given, record render timing.

    Returns
    -------
    np.ndarray
        BGR frame ready for display / virtual-camera output.
    """
    t0 = _now()
    mode = mode.lower().strip()
    if mode not in _DEBUG_MODES:
        mode = "blur"

    if mode == "blur":
        out = apply_background_blur(frame, mask, blur_strength, timer=timer)
    elif mode == "mask":
        out = _render_mask(frame, mask)
    elif mode == "edges":
        out = _render_edges(frame, mask)
    elif mode == "split":
        out = _render_split(frame, mask, blur_strength)
    elif mode == "heatmap":
        out = _render_heatmap(frame, raw_mask)
    else:  # original
        out = frame.copy()

    if fps is not None:
        out = _overlay_fps(out, fps)
    if timer:
        timer.mark("render", t0)

    return out


def resize_to_fit(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize frame to fit target dimensions while preserving aspect ratio.
    Pads with black bars to fill the target dimensions.
    """
    h, w = frame.shape[:2]
    if w == target_width and h == target_height:
        return frame

    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use sharper interpolation for upscaling, area for downscaling
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return result
