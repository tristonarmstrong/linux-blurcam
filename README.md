# blurcam

**Virtual webcam with AI background blur for Linux**

One command to blur your background on Discord, Zoom, Google Meet, or any video app.

GPU-accelerated via ONNX Runtime CUDA (falls back to CPU if GPU unavailable).
Works on x86_64 and ARM64 (Asahi Linux, Raspberry Pi, Ubuntu, Fedora, Arch, and more).

## Install

```bash
curl -sSL https://raw.githubusercontent.com/jolehuit/linux-blurcam/main/scripts/install.sh | bash
```

That's it. The installer:
- Installs [uv](https://docs.astral.sh/uv/) (fast Python manager) if needed
- Installs the v4l2loopback kernel module for your distro
- Creates the virtual camera at `/dev/video10`
- Starts the background daemon that auto-detects when apps use the camera

**GPU support:** By default, blurcam installs `onnxruntime-gpu` for NVIDIA CUDA acceleration.
If you need CPU-only (e.g., ARM64 without NVIDIA GPU), install with `blurcam[cpu]`.

**v4l2loopback requirement:** The virtual camera **must** be loaded with `exclusive_caps=1` or `pyvirtualcam` cannot write to it. The install script handles this, but if you do it manually, don't forget that flag.

## Usage

In your video app (Discord, Zoom, Meet...), select **"BlurCam"** as your camera.

That's it. **Blur starts automatically** when an app uses BlurCam, and stops when the app closes.

Your real webcam only turns on when needed — no manual commands required.

**Important:** If another app (e.g., Discord, a browser tab) already has your real webcam open, blurcam cannot access it. Close the camera in other apps first, or start blurcam before opening your video app.

### Adjust blur strength

```bash
# See current settings
blurcam config

# Change blur (applies immediately, no restart needed)
blurcam config --blur 45
```

Blur values: `15` = subtle, `35` = default, `55` = strong, `75` = very blurry

### Debug / preview modes

See exactly what the AI sees and tune the mask in real time.

```bash
# Open an interactive preview window
blurcam preview

# Inside the preview window:
#   [B]lur  [M]ask  [E]dges  [S]plit  [H]eatmap  [O]riginal
#   [F]PS toggle  [P]rofile toggle  [+/-] threshold  [UP/DOWN] blur strength
#   [Q]uit / Esc

# Change what the virtual camera sends (live, no restart)
blurcam config --debug mask
blurcam config --debug edges
blurcam config --show-fps true
```

| Mode | Description |
|------|-------------|
| `blur` | Default — blurred background |
| `mask` | Grayscale segmentation mask (tune threshold with +/-) |
| `edges` | Green contour around the detected foreground |
| `split` | Left = original, right = blurred |
| `heatmap` | Color confidence map (blue = low, red = high) |
| `original` | Unmodified webcam feed |

### FPS & profiling

If your framerate feels low, use profiling to find the bottleneck:

```bash
# Preview window with per-frame timing breakdown
blurcam preview --profile --show-fps

# Daemon mode: print timing stats every 3 seconds
blurcam --profile

# Enable profiling permanently
blurcam config --profile true
```

The profiler shows average milliseconds per stage:

```
FPS=28.3 capture=30.1ms | preprocess=1.2ms | inference=0.8ms | blur-op=8.5ms | blend=2.1ms
```

If `capture` is ~30ms, your **webcam** is the bottleneck (most USB cameras are hardware-limited to 30 FPS). If `blur-op` is high, reduce blur strength with `blurcam config --blur 15`.

### How to test

1. Open your video app (Discord, Zoom, Meet...)
2. Select **"BlurCam"** as your camera
3. You should see yourself with blurred background

## Known Limitations

### Switching cameras in Electron apps (Discord, Vesktop, etc.)

Due to how V4L2 works on Linux (only one process can use the webcam at a time), switching from BlurCam to your real camera may show a black screen.

**Workaround:** Close the camera settings menu, wait 1-2 seconds, then reopen it and select your camera.

This happens because Electron apps don't release the previous camera fast enough when switching. It's a timing issue between the app and the daemon.

### Switching segmentation models

blurcam supports two portrait segmentation models out of the box:

| Model | Input | Output | Size | Best for |
|-------|-------|--------|------|----------|
| `mediapipe` | 256×256 | 1-channel alpha | 452 KB | Default — fast, good edges |
| `sinet` | 320×320 | 2-class softmax | 427 KB | Slightly higher resolution |

```bash
# Switch to SINet (downloads automatically on first use)
blurcam config --model sinet

# Switch back to MediaPipe
blurcam config --model mediapipe

# Pre-download a model without running
blurcam --download-model
```

The change applies immediately — no restart needed. The model is auto-detected from its ONNX shape signature, so preprocessing and postprocessing adjust automatically.

### Webcam FPS ceiling

Most USB webcams are hardware-limited to **30 FPS**. Even on a powerful GPU, the `capture` stage will show ~30ms. This is expected — the AI model runs in < 2ms on an RTX 4060.

To get higher FPS, use a 60 FPS webcam (e.g., Logitech Brio, Razer Kiyo Pro).

## Uninstall

```bash
blurcam uninstall
```

This removes config files, cached model, and systemd service.

## Manual install

If you prefer to install manually:

### 1. Install v4l2loopback

**Fedora / Asahi Linux:**
```bash
sudo dnf install akmod-v4l2loopback
# Asahi only: sudo dnf install kernel-16k-devel && sudo akmods --force
```

**Ubuntu / Debian / Raspberry Pi OS:**
```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils
```

**Arch Linux:**
```bash
sudo pacman -S v4l2loopback-dkms
```

### 2. Load the kernel module

**Required:** `exclusive_caps=1` is mandatory — without it, the device only shows as "capture" and `pyvirtualcam` cannot write to it.

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="BlurCam" exclusive_caps=1
```

### 3. Install blurcam

**Default (GPU-accelerated with CUDA):**
```bash
# With uv (recommended)
uv tool install blurcam

# Or with pipx
pipx install blurcam

# Or with pip
pip install --user blurcam
```

**CPU-only (for ARM64 or systems without NVIDIA GPU):**
```bash
# With uv
uv tool install "blurcam[cpu]"

# Or with pipx
pipx install "blurcam[cpu]"

# Or with pip
pip install --user "blurcam[cpu]"
```

### 4. Run

```bash
blurcam
```

## Auto-start (default)

The installer automatically enables the daemon at boot. The daemon is lightweight — it only watches for app connections and starts the webcam when needed.

### Manual control

```bash
# Check status
systemctl --user status blurcam

# Stop daemon
systemctl --user stop blurcam

# Disable auto-start
systemctl --user disable blurcam
```

### v4l2loopback config (for manual install)

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=10 card_label="BlurCam" exclusive_caps=1' | \
    sudo tee /etc/modprobe.d/v4l2loopback.conf
```

## Troubleshooting

### "Virtual camera not found"

The kernel module isn't loaded:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="BlurCam" exclusive_caps=1
```

### "Could not open webcam"

Another app is using the camera. Close other video apps first.

### "Failed to create CUDAExecutionProvider"

Your NVIDIA drivers or cuDNN are missing or incompatible. blurcam will automatically fall back to CPU, but for GPU acceleration you need:

- NVIDIA GPU with driver 550+
- CUDA 12.x
- cuDNN 9.x

Check with:
```bash
nvidia-smi
ldconfig -p | grep libcudnn
```

### Pop!_OS / Ubuntu 24.04: cuDNN version mismatch

**Problem:** Pop!_OS 24.04 ships `nvidia-cudnn` 8.9.2, but `onnxruntime-gpu` 1.26+ requires **cuDNN 9.x**. You may see:

```
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*.
Failed to load library ... libcudnn.so.9: cannot open shared object file
```

**Solution:** Install cuDNN 9 manually from NVIDIA:

```bash
# 1. Download cuDNN 9 for CUDA 12 from NVIDIA
# https://developer.nvidia.com/cudnn-downloads
# Select: Linux → x86_64 → Ubuntu → 24.04 → deb (local)

# 2. Install the local repo package
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.x.x.x_1.0-1_amd64.deb

# 3. Import the GPG key
sudo cp /var/cudnn-local-repo-ubuntu2404-9.x.x.x/cudnn-local-9AE49B96-keyring.gpg /usr/share/keyrings/

# 4. Update apt and install cuDNN 9 for CUDA 12
sudo apt update
sudo apt install cudnn-cuda-12

# 5. Verify
ldconfig -p | grep libcudnn.so.9
# Should show: libcudnn.so.9 => /usr/lib/x86_64-linux-gnu/libcudnn.so.9
```

**Version compatibility reference:**

| Component | Pop!_OS 24.04 Default | Required by ORT 1.26 | Status |
|-----------|----------------------|----------------------|--------|
| CUDA | 12.0 (via `nvidia-cuda-toolkit`) | 12.x | ✅ OK |
| cuDNN | 8.9.2 (via `nvidia-cudnn`) | 9.x | ❌ Need manual install |
| GPU Driver | 580.x | 550+ | ✅ OK |

**Note:** We also pin `numpy<2` in the package dependencies because `onnxruntime-gpu` 1.18.x (the CUDA 11 build) doesn't support NumPy 2.x. The current package uses `numpy>=1.26.0,<2` for compatibility.

### Check system status

```bash
blurcam-setup
```

### Low FPS

**Step 1: Check if the webcam is the bottleneck**
```bash
blurcam preview --profile
```
If `capture` is ~30ms, your webcam is capped at 30 FPS. This is normal.

**Step 2: Reduce blur strength**
```bash
blurcam config --blur 15
```

**Step 3: Check GPU is active**
```bash
blurcam config
# Look for: ONNX Runtime using GPU: CUDAExecutionProvider
```
If it says CPU, see the CUDA/cuDNN requirements above.

## Advanced options

```bash
# Detection sensitivity (if edges are cut off or background leaks through)
blurcam config --threshold 0.4   # More inclusive (keeps more of you)
blurcam config --threshold 0.6   # More strict (cleaner edges)

# Use different webcam
blurcam --input 1

# Use different virtual camera
blurcam --output /dev/video20

# Run with debug output (mask, edges, heatmap, etc.)
blurcam --debug mask
blurcam --show-fps

# Run with profiling (shows timing every 3 seconds)
blurcam --profile

# All config settings (apply live, no restart)
blurcam config --blur 35
blurcam config --threshold 0.5
blurcam config --debug blur
blurcam config --show-fps true
blurcam config --profile true
blurcam config --model mediapipe
```

## How it works

1. **Daemon** watches the virtual camera device using inotify
2. When an app opens BlurCam, the daemon starts processing
3. **ONNX Runtime** runs the AI segmentation model on **GPU** (CUDA) when available, falling back to CPU
4. **OpenCV** captures your webcam and applies background blur
5. **pyvirtualcam** + **v4l2loopback** sends the result to the virtual camera
6. When the app closes, the daemon stops processing (webcam turns off)

The key: using the ONNX model directly instead of MediaPipe Python bindings (which don't support ARM64), with GPU acceleration for maximum FPS.

**Daemon behavior:**
- The daemon is idle (sending black frames) when no app is using BlurCam
- When an app opens BlurCam, the daemon lazily loads the AI model and opens the webcam
- Config changes (`blurcam config --blur 45`, `--model sinet`, etc.) apply immediately without restart
- When the app closes BlurCam, the daemon releases the webcam and model instantly

### Model details

**MediaPipe** (default):

| Spec | Value |
|------|-------|
| **Model** | MediaPipe Selfie Segmentation (ONNX export) |
| **Source** | HuggingFace `onnx-community/mediapipe_selfie_segmentation` |
| **Size** | 452 KB |
| **Input** | 256×256 RGB (NCHW float) |
| **Output** | 256×256 alpha mask (0–1 confidence) |
| **Architecture** | Lightweight CNN (MobileNet backbone) |
| **GPU inference** | < 2ms on RTX 4060 |

**SINet** (optional):

| Spec | Value |
|------|-------|
| **Model** | SINet (Portrait Segmentation) |
| **Source** | GitHub `anilsathyan7/Portrait-Segmentation` |
| **Size** | 427 KB |
| **Input** | 320×320 BGR (NCHW float, ImageNet normalized) |
| **Output** | 320×320 2-class softmax (foreground/background) |
| **Architecture** | SINet with Information Blocking Decoder + Spatial Squeeze |
| **GPU inference** | < 2ms on RTX 4060 |

Both models are tiny enough that the **webcam is usually the bottleneck**, not the GPU.

## Compatibility

| Platform | Status |
|----------|--------|
| x86_64 Linux (NVIDIA CUDA) | Tested |
| Asahi Linux (M1/M2/M3) | Tested |
| Raspberry Pi 5 | Not tested yet |
| Raspberry Pi 4 | Not tested yet |
| Ubuntu ARM64 | Not tested yet |
| Fedora ARM64 | Not tested yet |
| Arch ARM | Not tested yet |

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- [ONNX Community](https://huggingface.co/onnx-community) - Exported segmentation model
- [Astral](https://astral.sh/) - uv package manager
- [v4l2loopback](https://github.com/umlaeute/v4l2loopback) - Virtual camera kernel module
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) - Python virtual camera library
