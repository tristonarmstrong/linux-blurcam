#!/usr/bin/env python3
"""
blurcam - Virtual webcam with AI background blur for Linux

GPU-accelerated via ONNX Runtime CUDA (falls back to CPU if GPU unavailable).

Usage:
    blurcam                    Run the daemon (auto-detects when apps use BlurCam)
    blurcam config             Show current settings
    blurcam config --blur 45   Adjust blur strength (live)
    blurcam uninstall          Remove blurcam completely
"""

import argparse
import os
import sys
import time

from . import __version__
from .config import load_config, update_config, get_config_mtime, DEFAULT_CONFIG


def cmd_uninstall(args):
    """Uninstall blurcam completely."""
    import shutil
    import subprocess
    from pathlib import Path

    print("Uninstalling blurcam...\n")

    # 1. Stop and disable systemd service
    print("Stopping service...")
    subprocess.run(
        ["systemctl", "--user", "stop", "blurcam"],
        capture_output=True
    )
    subprocess.run(
        ["systemctl", "--user", "disable", "blurcam"],
        capture_output=True
    )

    # 2. Remove systemd service file
    print("Removing systemd service...")
    service_file = Path.home() / ".config/systemd/user/blurcam.service"
    if service_file.exists():
        service_file.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)

    # 3. Remove config files
    print("Removing config files...")
    config_dir = Path.home() / ".config/blurcam"
    if config_dir.exists():
        shutil.rmtree(config_dir)

    # 4. Remove cached model
    print("Removing cached model...")
    cache_dir = Path.home() / ".cache/blurcam"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    # 5. Remove uv tool environment (onnx, opencv, etc.)
    print("Removing Python environment...")
    uv_tool_dir = Path.home() / ".local/share/uv/tools/blurcam"
    if uv_tool_dir.exists():
        shutil.rmtree(uv_tool_dir)

    # 6. Remove shell aliases
    print("Removing shell aliases...")
    for rc_name in [".bashrc", ".zshrc"]:
        rc_file = Path.home() / rc_name
        if rc_file.exists():
            content = rc_file.read_text()
            # Remove the alias block
            import re
            new_content = re.sub(
                r'\n# blurcam shortcuts\n.*?alias noblur=.*?\n',
                '\n',
                content,
                flags=re.DOTALL
            )
            if new_content != content:
                rc_file.write_text(new_content)

    # 7. Remove v4l2loopback config (requires sudo)
    print("Removing v4l2loopback config...")
    v4l_configs = [
        Path("/etc/modprobe.d/v4l2loopback.conf"),
        Path("/etc/modules-load.d/v4l2loopback.conf"),
    ]
    for conf in v4l_configs:
        if conf.exists():
            result = subprocess.run(
                ["sudo", "rm", "-f", str(conf)],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"  Run manually: sudo rm {conf}")

    # 8. Uninstall the tool itself via uv
    print("Removing blurcam...")
    subprocess.run(
        ["uv", "tool", "uninstall", "blurcam"],
        capture_output=True
    )

    print("\n✓ blurcam uninstalled!\n")
    return 0


def cmd_config(args):
    """Handle config subcommand."""
    # Build list of fields to update
    update_fields = ["blur", "threshold", "debug", "show_fps", "profile", "model"]
    has_update = any(getattr(args, f, None) is not None for f in update_fields)

    if has_update:
        updates = {}
        if args.blur is not None:
            updates["blur"] = args.blur if args.blur % 2 == 1 else args.blur + 1
        if args.threshold is not None:
            updates["threshold"] = args.threshold
        if args.debug is not None:
            updates["debug"] = args.debug
        if args.show_fps is not None:
            updates["show_fps"] = args.show_fps
        if args.profile is not None:
            updates["profile"] = args.profile
        if args.model is not None:
            updates["model"] = args.model

        config = update_config(**updates)
        print("Settings updated:")
        for k in updates:
            print(f"  {k}: {config[k]}")
        print("Changes apply immediately to running instance.")
        return 0

    # Otherwise show current config
    config = load_config()
    print("Current settings:")
    print(f"  blur: {config['blur']}")
    print(f"  threshold: {config['threshold']} (detection sensitivity, 0-1)")
    print(f"  debug: {config['debug']} (blur | mask | edges | split | heatmap | original)")
    print(f"  show_fps: {config['show_fps']}")
    print(f"  profile: {config.get('profile', False)}")
    print(f"  model: {config.get('model', 'mediapipe')}")
    print()
    print("Adjust with:")
    print("  blurcam config --blur 45")
    print("  blurcam config --debug mask")
    print("  blurcam config --show-fps true")
    print("  blurcam config --profile true")
    print("  blurcam config --model sinet")
    return 0


def cmd_run(args):
    """Run the virtual camera."""
    # Lazy imports to avoid ONNX warnings on --help
    import cv2
    import numpy as np
    import pyvirtualcam
    from .models import get_model_path
    from .segmentation import SelfieSegmentation, render_frame
    from .config import get_config_path

    # Load config
    config = load_config()

    # CLI args override config
    if args.blur is not None:
        config["blur"] = args.blur if args.blur % 2 == 1 else args.blur + 1
    if args.threshold is not None:
        config["threshold"] = args.threshold
    if args.input is not None:
        config["input"] = args.input
    if args.output is not None:
        config["output"] = args.output
    if args.debug is not None:
        config["debug"] = args.debug
    if args.show_fps is not None:
        config["show_fps"] = args.show_fps

    # Ensure blur is odd
    blur_strength = config["blur"] if config["blur"] % 2 == 1 else config["blur"] + 1
    threshold = config["threshold"]
    debug_mode = config["debug"]
    show_fps = config["show_fps"]

    # Get/download model
    print("Loading model...")
    model_path = get_model_path(model_name=config.get("model", "mediapipe"))
    segmentation = SelfieSegmentation(model_path)
    print(f"Model loaded: {model_path}")
    active_providers = segmentation.session.get_providers()
    if active_providers:
        print(f"Execution provider: {active_providers[0]}")

    # Open webcam
    print(f"Opening webcam {config['input']}...")
    cap = cv2.VideoCapture(config["input"])
    # Request MJPG: uncompressed YUYV saturates USB bandwidth and caps
    # high-res capture at a few fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
    cap.set(cv2.CAP_PROP_FPS, config["fps"])

    if not cap.isOpened():
        print(f"Error: Could not open webcam {config['input']}", file=sys.stderr)
        print("Make sure no other application is using the webcam.", file=sys.stderr)
        return 1

    # Get actual dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Webcam opened: {actual_width}x{actual_height} @ {actual_fps}fps")

    # Check virtual camera device
    if not os.path.exists(config["output"]):
        print(f"\nError: Virtual camera {config['output']} not found!", file=sys.stderr)
        print("Run 'blurcam-setup' to configure v4l2loopback.", file=sys.stderr)
        cap.release()
        return 1

    print(f"Opening virtual camera {config['output']}...")
    print(f"Debug mode: {debug_mode} (use 'blurcam config --debug <mode>' to change)")

    try:
        with pyvirtualcam.Camera(
            width=actual_width,
            height=actual_height,
            fps=config["fps"],
            device=config["output"],
            fmt=pyvirtualcam.PixelFormat.RGB,
        ) as vcam:
            print(f"Virtual camera started: {vcam.device}")
            print(f"Blur: {blur_strength}")
            print()
            print("Press Ctrl+C to stop")
            print("Adjust blur: blurcam config --blur 45")
            print()

            frame_count = 0
            start_time = time.time()
            last_config_check = time.time()
            config_mtime = get_config_mtime()

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame", file=sys.stderr)
                    break

                # Check for config changes every second
                now = time.time()
                if now - last_config_check >= 1.0:
                    new_mtime = get_config_mtime()
                    if new_mtime > config_mtime:
                        config_mtime = new_mtime
                        new_config = load_config()
                        new_blur = new_config["blur"] if new_config["blur"] % 2 == 1 else new_config["blur"] + 1
                        if new_blur != blur_strength or new_config["threshold"] != threshold:
                            blur_strength = new_blur
                            threshold = new_config["threshold"]
                            print(f"\rSettings updated: blur={blur_strength}  ", flush=True)
                        if new_config.get("debug") != debug_mode:
                            debug_mode = new_config["debug"]
                            print(f"\rDebug mode changed: {debug_mode}  ", flush=True)
                        if new_config.get("show_fps") != show_fps:
                            show_fps = new_config["show_fps"]
                    last_config_check = now

                # Get raw mask and binary mask
                raw_mask = segmentation.predict(frame)
                mask = segmentation.get_mask(frame, threshold=threshold)

                # Compute FPS for overlay
                fps_val = None
                if show_fps or debug_mode != "blur":
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps_val = frame_count / elapsed

                # Render with debug mode
                result = render_frame(
                    frame, mask, raw_mask,
                    mode=debug_mode,
                    blur_strength=blur_strength,
                    fps=fps_val if show_fps else None,
                )

                # Convert BGR to RGB for virtual camera
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                # Send to virtual camera
                vcam.send(result_rgb)
                vcam.sleep_until_next_frame()

                # Print FPS every second
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    print(f"\rFPS: {fps:.1f}  ", end="", flush=True)
                    frame_count = 0
                    start_time = time.time()

    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    finally:
        cap.release()

    return 0


def cmd_preview(args):
    """Open a live preview window with debug overlays."""
    import cv2
    import numpy as np
    from .models import get_model_path
    from .segmentation import SelfieSegmentation, render_frame, FrameTimer

    # Load config
    config = load_config()

    # CLI overrides
    if args.blur is not None:
        config["blur"] = args.blur if args.blur % 2 == 1 else args.blur + 1
    if args.threshold is not None:
        config["threshold"] = args.threshold
    if args.input is not None:
        config["input"] = args.input

    blur_strength = config["blur"] if config["blur"] % 2 == 1 else config["blur"] + 1
    threshold = config["threshold"]
    debug_mode = args.debug if args.debug else config["debug"]
    show_fps = args.show_fps if args.show_fps is not None else config["show_fps"]
    show_profile = args.profile if hasattr(args, 'profile') and args.profile else False

    print("Loading model...")
    model_path = get_model_path(model_name=config.get("model", "mediapipe"))
    segmentation = SelfieSegmentation(model_path)
    print(f"Model loaded: {model_path}")
    active_providers = segmentation.session.get_providers()
    if active_providers:
        print(f"Execution provider: {active_providers[0]}")

    print(f"Opening webcam {config['input']}...")
    cap = cv2.VideoCapture(config["input"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
    cap.set(cv2.CAP_PROP_FPS, config["fps"])

    if not cap.isOpened():
        print(f"Error: Could not open webcam {config['input']}", file=sys.stderr)
        return 1

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {actual_width}x{actual_height}")
    print()
    print("Controls:")
    print("  [B]lur  [M]ask  [E]dges  [S]plit  [H]eatmap  [O]riginal")
    print("  [F]PS toggle  [P]rofile toggle  [+/-] threshold  [UP/DOWN] blur")
    print("  [Q]uit / Ctrl+C")
    print()

    # Check if OpenCV GUI backend is available (headless builds won't have it)
    headless = False
    try:
        cv2.namedWindow("blurcam preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("blurcam preview", actual_width, actual_height)
    except cv2.error as e:
        if "not implemented" in str(e).lower() or "namedwindow" in str(e).lower():
            headless = True
            print("Warning: OpenCV was built without GUI support (headless).")
            print("  Preview window unavailable.")
            print("  Install 'opencv-python' (not headless) for preview.")
            print("  Continuing in text-only mode. Press Ctrl+C to stop.")
            print()
        else:
            raise

    frame_count = 0
    start_time = time.time()
    fps_val = 0.0
    timer = FrameTimer(window=60)
    profile_counter = 0

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if timer:
                timer.mark("capture", t0)
            if not ret:
                print("Error reading frame", file=sys.stderr)
                break

            # Segmentation
            raw_mask = segmentation.predict(frame, timer=timer)
            mask = segmentation.get_mask(frame, threshold=threshold, timer=timer)

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps_val = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
                profile_counter += 1
                if profile_counter % 3 == 0 and show_profile:
                    print(f"\r{timer.report(f'FPS={fps_val:.1f}')}  ", end="", flush=True)
                    profile_counter = 0
                if headless and not show_profile:
                    print(f"\rFPS={fps_val:.1f}  mode={debug_mode}  thr={threshold:.2f}  blur={blur_strength}  ", end="", flush=True)

            # Render
            result = render_frame(
                frame, mask, raw_mask,
                mode=debug_mode,
                blur_strength=blur_strength,
                fps=fps_val if show_fps else None,
                timer=timer,
            )

            if not headless:
                # Overlay info text
                info_text = f"mode={debug_mode} thr={threshold:.2f} blur={blur_strength}"
                if show_fps:
                    info_text += f" fps={fps_val:.1f}"
                if show_profile:
                    info_text = timer.report(info_text)
                cv2.putText(result, info_text, (10, actual_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow("blurcam preview", result)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('b'):
                    debug_mode = "blur"
                elif key == ord('m'):
                    debug_mode = "mask"
                elif key == ord('e'):
                    debug_mode = "edges"
                elif key == ord('s'):
                    debug_mode = "split"
                elif key == ord('h'):
                    debug_mode = "heatmap"
                elif key == ord('o'):
                    debug_mode = "original"
                elif key == ord('f'):
                    show_fps = not show_fps
                elif key == ord('p'):
                    show_profile = not show_profile
                    timer.reset()
                    print(f"\nProfile overlay: {'on' if show_profile else 'off'}")
                elif key == ord('+') or key == ord('='):
                    threshold = min(1.0, threshold + 0.05)
                elif key == ord('-') or key == ord('_'):
                    threshold = max(0.0, threshold - 0.05)
                elif key == 82:  # UP arrow
                    blur_strength += 2
                elif key == 84:  # DOWN arrow
                    blur_strength = max(1, blur_strength - 2)
            else:
                # Headless mode: just sleep briefly so we don't spin the CPU
                import time as _time
                _time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"\nFinal settings: threshold={threshold:.2f} blur={blur_strength}")
        print(f"Save with: blurcam config --threshold {threshold:.2f} --blur {blur_strength}")

    return 0


def cmd_daemon(args):
    """Run in daemon mode - auto-starts blur when BlurCam is accessed."""
    from .daemon import run_daemon
    from .config import load_config, update_config

    config = load_config()
    input_device = args.input if args.input is not None else config["input"]
    output_device = args.output if args.output is not None else config["output"]

    # If CLI overrides debug/show_fps/profile, persist them so daemon picks them up
    overrides = {}
    if args.debug is not None:
        overrides["debug"] = args.debug
    if args.show_fps is not None:
        overrides["show_fps"] = args.show_fps
    if getattr(args, "profile", None) is not None:
        overrides["profile"] = args.profile
    if overrides:
        update_config(**overrides)

    profile = getattr(args, "profile", None)
    if profile is None:
        profile = config.get("profile", False)

    return run_daemon(
        input_device=input_device,
        output_device=output_device,
        profile=profile,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Virtual webcam with AI background blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  blurcam                    Run daemon (auto-starts when BlurCam is used)
  blurcam config --blur 45   Adjust blur strength
  blurcam config             Show current settings
  blurcam preview            Open live preview window with debug modes
  blurcam uninstall          Uninstall completely

The daemon watches BlurCam - blur starts automatically when an app uses it.
        """,
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"blurcam {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Config subcommand
    config_parser = subparsers.add_parser("config", help="View or adjust settings")
    config_parser.add_argument(
        "--blur", "-b",
        type=int,
        help="Blur strength (odd number, default: 35)",
    )
    config_parser.add_argument(
        "--threshold", "-t",
        type=float,
        help="Detection sensitivity 0-1 (advanced, default: 0.5)",
    )
    config_parser.add_argument(
        "--debug",
        choices=["blur", "mask", "edges", "split", "heatmap", "original"],
        help="Output mode for virtual camera (default: blur)",
    )
    config_parser.add_argument(
        "--show-fps",
        choices=["true", "false"],
        help="Overlay FPS counter on output (default: false)",
    )
    config_parser.add_argument(
        "--profile",
        choices=["true", "false"],
        help="Print per-frame timing breakdown every 3 seconds (default: false)",
    )
    config_parser.add_argument(
        "--model",
        choices=["mediapipe", "sinet"],
        help="Segmentation model to use (default: mediapipe)",
    )

    # Preview subcommand
    preview_parser = subparsers.add_parser("preview", help="Open live preview window")
    preview_parser.add_argument(
        "--input", "-i", type=int,
        help="Input webcam device number",
    )
    preview_parser.add_argument(
        "--blur", "-b", type=int,
        help="Blur strength (odd number, default: 35)",
    )
    preview_parser.add_argument(
        "--threshold", "-t", type=float,
        help="Detection sensitivity 0-1 (default: 0.5)",
    )
    preview_parser.add_argument(
        "--debug",
        choices=["blur", "mask", "edges", "split", "heatmap", "original"],
        default="blur",
        help="Initial preview mode (default: blur)",
    )
    preview_parser.add_argument(
        "--show-fps", action="store_true",
        help="Overlay FPS counter",
    )
    preview_parser.add_argument(
        "--profile", action="store_true",
        help="Overlay per-frame timing breakdown (capture|preprocess|inference|blur|...)",
    )

    # Uninstall subcommand
    subparsers.add_parser("uninstall", help="Uninstall blurcam completely")

    # Daemon arguments (on main parser since it's the default)
    parser.add_argument(
        "--input", "-i",
        type=int,
        help="Input webcam device number",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output virtual camera device",
    )
    parser.add_argument(
        "--debug",
        choices=["blur", "mask", "edges", "split", "heatmap", "original"],
        help="Output mode (default: blur)",
    )
    parser.add_argument(
        "--show-fps", action="store_true",
        help="Overlay FPS counter on output",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print per-frame timing breakdown every 3 seconds (stdout)",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model and exit",
    )

    args = parser.parse_args()

    # Convert string "true"/"false" to bool for config subcommand
    if hasattr(args, "show_fps") and isinstance(args.show_fps, str):
        args.show_fps = args.show_fps.lower() == "true"
    if hasattr(args, "profile") and isinstance(args.profile, str):
        args.profile = args.profile.lower() == "true"

    # Download model only
    if getattr(args, 'download_model', False):
        from .models import get_model_path
        from .config import load_config
        cfg = load_config()
        model_name = cfg.get("model", "mediapipe")
        get_model_path(model_name=model_name, force_download=True)
        print(f"Model '{model_name}' downloaded successfully!")
        return 0

    # Route to subcommand
    if args.command == "config":
        return cmd_config(args)
    elif args.command == "uninstall":
        return cmd_uninstall(args)
    elif args.command == "preview":
        return cmd_preview(args)
    else:
        return cmd_daemon(args)


if __name__ == "__main__":
    sys.exit(main())
