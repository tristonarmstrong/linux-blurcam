"""
Daemon mode for blurcam - always produces frames, blur activates when consumers connect.
Uses inotify for instant detection of consumer connect/disconnect.
"""

import os
import signal
import sys
import threading
import time

import numpy as np
from inotify_simple import INotify, flags


class BlurDaemon:
    """Always writes to virtual cam. Blur activates when consumers connect."""

    def __init__(self, device="/dev/video10", input_device=0, profile=False):
        self.device = device
        self.input_device = input_device
        self.profile = profile
        self.blur_active = False
        self.stop_event = threading.Event()
        self.consumer_event = threading.Event()  # Set when consumer state changes
        self.has_consumers = False

        # Load config
        from .config import load_config
        self.config = load_config()

        self.width = self.config["width"]
        self.height = self.config["height"]
        self.fps = self.config["fps"]

    def _get_consumer_count(self):
        """Count processes that have the device open (excluding ourselves)."""
        my_pid = os.getpid()
        count = 0

        try:
            for pid_dir in os.listdir('/proc'):
                if not pid_dir.isdigit():
                    continue
                pid = int(pid_dir)
                if pid == my_pid:
                    continue
                try:
                    fd_dir = f'/proc/{pid}/fd'
                    for fd in os.listdir(fd_dir):
                        try:
                            target = os.readlink(f'{fd_dir}/{fd}')
                            if target == self.device:
                                count += 1
                                break
                        except (OSError, PermissionError):
                            pass
                except (OSError, PermissionError):
                    pass
        except Exception:
            pass

        return count

    def _inotify_watcher(self):
        """Watch for open/close events on the virtual camera device."""
        inotify = INotify()
        watch_flags = flags.OPEN | flags.CLOSE_NOWRITE | flags.CLOSE_WRITE

        try:
            wd = inotify.add_watch(self.device, watch_flags)
        except Exception as e:
            print(f"Warning: Could not watch {self.device}: {e}", file=sys.stderr, flush=True)
            return

        while not self.stop_event.is_set():
            # Read with timeout so we can check stop_event
            events = inotify.read(timeout=500)

            for event in events:
                if event.mask & flags.OPEN:
                    # Someone opened the device - check if it's a real consumer
                    time.sleep(0.05)  # Brief delay for fd to be registered
                    consumers = self._get_consumer_count()
                    if consumers > 0:
                        self.has_consumers = True
                        self.consumer_event.set()

                elif event.mask & (flags.CLOSE_NOWRITE | flags.CLOSE_WRITE):
                    # Someone closed the device - check if consumers remain
                    time.sleep(0.05)  # Brief delay for fd to be unregistered
                    consumers = self._get_consumer_count()
                    if consumers == 0:
                        self.has_consumers = False
                        self.consumer_event.set()

        inotify.close()

    def run(self):
        """Main daemon loop - always writes frames."""
        import cv2
        import pyvirtualcam
        from .models import get_model_path
        from .segmentation import SelfieSegmentation, render_frame, FrameTimer, resize_to_fit
        from .config import load_config, get_config_mtime

        print(f"blurcam daemon started", flush=True)
        print(f"Virtual camera: {self.device}", flush=True)
        print(f"Press Ctrl+C to stop", flush=True)
        print(flush=True)

        # Check if device exists
        if not os.path.exists(self.device):
            print(f"Error: {self.device} not found", file=sys.stderr, flush=True)
            print("Run 'blurcam-setup' to configure v4l2loopback.", file=sys.stderr, flush=True)
            return 1

        # Handle signals
        def signal_handler(sig, frame):
            print("\nShutting down...", flush=True)
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start inotify watcher thread
        watcher_thread = threading.Thread(target=self._inotify_watcher, daemon=True)
        watcher_thread.start()

        # Lazy-load model and webcam (only when needed)
        segmentation = None
        cap = None
        model_path = None

        # Create black frame for idle mode
        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        config_mtime = get_config_mtime()
        blur_strength = self.config["blur"]
        if blur_strength % 2 == 0:
            blur_strength += 1
        threshold = self.config["threshold"]
        debug_mode = self.config.get("debug", "blur")
        show_fps = self.config.get("show_fps", False)
        current_model = self.config.get("model", "mediapipe")

        frame_count = 0
        start_time = time.time()
        fps_val = 0.0
        timer = FrameTimer(window=60)
        profile_counter = 0

        try:
            with pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self.device,
                fmt=pyvirtualcam.PixelFormat.RGB,
            ) as vcam:
                print(f"Virtual camera ready: {vcam.device}", flush=True)
                print(f"Select 'BlurCam' in your video app", flush=True)
                print(f"Debug mode: {debug_mode}", flush=True)
                print(flush=True)

                while not self.stop_event.is_set():
                    # Check for consumer state changes (non-blocking)
                    if self.consumer_event.is_set():
                        self.consumer_event.clear()

                        if self.has_consumers and not self.blur_active:
                            # Consumer connected - start blur
                            print(f"Consumer connected - starting blur", flush=True)
                            self.blur_active = True

                            # Load model if needed
                            if segmentation is None:
                                model_path = get_model_path(model_name=self.config.get("model", "mediapipe"))
                                segmentation = SelfieSegmentation(model_path)
                                print(f"Model loaded from {model_path}", flush=True)
                                current_model = self.config.get("model", "mediapipe")

                            # Open webcam
                            if cap is None or not cap.isOpened():
                                cap = cv2.VideoCapture(self.input_device)
                                # Request MJPG: uncompressed YUYV saturates USB
                                # bandwidth and caps high-res capture at a few fps.
                                cap.set(
                                    cv2.CAP_PROP_FOURCC,
                                    cv2.VideoWriter_fourcc(*"MJPG"),
                                )
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                cap.set(cv2.CAP_PROP_FPS, self.fps)

                                if not cap.isOpened():
                                    print(f"Error: Could not open webcam", file=sys.stderr, flush=True)
                                    self.blur_active = False
                                else:
                                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    print(f"Webcam actual resolution: {actual_width}x{actual_height}", flush=True)

                        elif not self.has_consumers and self.blur_active:
                            # No consumers - stop blur IMMEDIATELY
                            print(f"No consumers - releasing webcam", flush=True)
                            self.blur_active = False

                            # Release webcam immediately
                            if cap is not None:
                                cap.release()
                                cap = None

                    # Check for config changes
                    new_mtime = get_config_mtime()
                    if new_mtime > config_mtime:
                        config_mtime = new_mtime
                        self.config = load_config()
                        blur_strength = self.config["blur"]
                        if blur_strength % 2 == 0:
                            blur_strength += 1
                        threshold = self.config["threshold"]
                        debug_mode = self.config.get("debug", "blur")
                        show_fps = self.config.get("show_fps", False)

                        # Hot-swap model if changed
                        new_model = self.config.get("model", "mediapipe")
                        if new_model != current_model:
                            print(f"Model config changed: {current_model} -> {new_model}", flush=True)
                            current_model = new_model
                            if segmentation is not None:
                                segmentation = None
                                print(f"Model will reload on next consumer connect", flush=True)

                    # Generate frame
                    if self.blur_active and cap is not None and cap.isOpened():
                        t_capture = time.perf_counter()
                        ret, frame = cap.read()
                        if timer:
                            timer.mark("capture", t_capture)
                        if ret:
                            # Ensure frame matches expected dimensions (webcam may not support requested resolution)
                            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                                frame = resize_to_fit(frame, self.width, self.height)

                            # Segmentation
                            raw_mask = segmentation.predict(frame, timer=timer)
                            mask = segmentation.get_mask(frame, threshold=threshold, timer=timer)

                            # FPS counter
                            frame_count += 1
                            elapsed = time.time() - start_time
                            if elapsed >= 1.0:
                                fps_val = frame_count / elapsed
                                frame_count = 0
                                start_time = time.time()
                                # Print profiling every 3 seconds if enabled
                                if self.profile:
                                    profile_counter += 1
                                    if profile_counter % 3 == 0:
                                        print(f"\r{timer.report(f'FPS={fps_val:.1f}')}  ", end="", flush=True)
                                        profile_counter = 0

                            result = render_frame(
                                frame, mask, raw_mask,
                                mode=debug_mode,
                                blur_strength=blur_strength,
                                fps=fps_val if show_fps else None,
                                timer=timer,
                            )
                            t_convert = time.perf_counter()
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            if timer:
                                timer.mark("cvtColor", t_convert)
                            t_send = time.perf_counter()
                            vcam.send(result_rgb)
                            if timer:
                                timer.mark("send", t_send)
                        else:
                            vcam.send(black_frame)
                    else:
                        # Send black frame when idle
                        vcam.send(black_frame)

                    vcam.sleep_until_next_frame()

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)
            return 1
        finally:
            if cap is not None:
                cap.release()

        return 0


def run_daemon(input_device=0, output_device="/dev/video10", profile=False):
    """Entry point for daemon mode."""
    daemon = BlurDaemon(device=output_device, input_device=input_device, profile=profile)
    return daemon.run()
