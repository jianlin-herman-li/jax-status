#!/usr/bin/env python3
"""Verbose JAX runtime inspection script."""

import sys
import os
import subprocess
import ctypes

# Preload CUDA driver library on Linux before importing JAX
# This ensures the library is available when JAX tries to initialize CUDA
if sys.platform == "linux":
    cuda_driver_lib = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    if os.path.exists(cuda_driver_lib):
        try:
            # Preload the CUDA driver library
            ctypes.CDLL(cuda_driver_lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass
        # Also set LD_LIBRARY_PATH as fallback
        cuda_driver_path = "/usr/lib/x86_64-linux-gnu"
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda_driver_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{current_ld_path}:{cuda_driver_path}" if current_ld_path else cuda_driver_path

import jax
import jax.lib


def main():
    """Main entry point for jax-status."""
    print("=" * 80)
    print("JAX Status Inspection")
    print("=" * 80)
    print()

    # System information
    print("API: sys.version")
    print(f"Value: {sys.version}")
    print()

    print("API: sys.executable")
    print(f"Value: {sys.executable}")
    print()

    # NVIDIA-SMI output
    print("API: subprocess.getoutput('nvidia-smi')")
    try:
        nvidia_smi_output = subprocess.getoutput("nvidia-smi")
        print(f"Value:\n{nvidia_smi_output}")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    # JAX version information
    print("API: jax.__version__")
    print(f"Value: {jax.__version__}")
    print()

    print("API: jax.lib.__version__")
    try:
        print(f"Value: {jax.lib.__version__}")
    except AttributeError:
        print("Value: AttributeError - jax.lib.__version__ not available")
    print()

    # JAX backend information
    print("API: jax.default_backend()")
    try:
        backend = jax.default_backend()
        print(f"Value: {backend}")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    # JAX devices
    print("API: jax.devices()")
    try:
        devices = jax.devices()
        print(f"Value: {devices}")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    print("API: jax.device_count()")
    try:
        count = jax.device_count()
        print(f"Value: {count}")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    print("API: jax.local_devices()")
    try:
        local_devices = jax.local_devices()
        print(f"Value: {local_devices}")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    # XLA bridge backend information
    print("API: jax.lib.xla_bridge.get_backend().platform")
    try:
        backend = jax.lib.xla_bridge.get_backend()
        platform = backend.platform
        print(f"Value: {platform}")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    print("API: jax.lib.xla_bridge.get_backend().platform_version")
    try:
        backend = jax.lib.xla_bridge.get_backend()
        platform_version = backend.platform_version
        print(f"Value: {platform_version}")
    except AttributeError:
        print("Value: AttributeError - platform_version not available")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    # CUDA-related metadata
    print("=" * 80)
    print("CUDA-related metadata")
    print("=" * 80)
    print()

    # Check if CUDA is available
    cuda_available = False
    try:
        backend = jax.lib.xla_bridge.get_backend()
        if backend.platform == "gpu":
            cuda_available = True
    except Exception:
        pass

    print("API: CUDA availability check")
    if cuda_available:
        print("Value: CUDA is available")
    else:
        print("Value: CUDA is NOT available")
    print()

    # GPU visibility check
    print("API: GPU visibility to JAX")
    try:
        devices = jax.devices()
        # Check for GPU devices - CUDA devices have device_kind "gpu" or platform "gpu"
        backend = jax.lib.xla_bridge.get_backend()
        is_gpu_backend = backend.platform == "gpu"
        gpu_devices = [d for d in devices if d.device_kind == "gpu" or (is_gpu_backend and "cuda" in str(d).lower())]
        if gpu_devices or is_gpu_backend:
            if gpu_devices:
                print(f"Value: JAX reports {len(gpu_devices)} GPU device(s): {gpu_devices}")
            else:
                # If backend is GPU but device_kind check didn't work, check device strings
                gpu_like = [d for d in devices if "cuda" in str(d).lower() or "gpu" in str(d).lower()]
                if gpu_like:
                    print(f"Value: JAX reports {len(gpu_like)} GPU device(s): {gpu_like}")
                else:
                    print(f"Value: JAX reports GPU backend with {len(devices)} device(s): {devices}")
        else:
            print("Value: JAX reports NO GPU devices")
            # Check if nvidia-smi shows GPUs
            try:
                nvidia_smi_output = subprocess.getoutput("nvidia-smi")
                if "NVIDIA-SMI" in nvidia_smi_output:
                    print("WARNING: nvidia-smi reports GPUs, but JAX does not see them!")
                    print("This indicates a potential configuration issue.")
            except Exception:
                pass
    except Exception as e:
        print(f"Value: Error checking GPU visibility - {e}")
    print()

    # Backend and platform name
    print("API: Backend and platform name")
    try:
        backend = jax.lib.xla_bridge.get_backend()
        print(f"Value: Platform = {backend.platform}")
        try:
            print(f"       Platform version = {backend.platform_version}")
        except AttributeError:
            pass
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    # CUDA version if available
    print("API: CUDA version")
    try:
        if hasattr(jax.lib, "xla_extension"):
            # Try to get CUDA version from JAX
            try:
                cuda_version = jax.lib.xla_extension.cuda_version()
                print(f"Value: {cuda_version}")
            except (AttributeError, Exception):
                print("Value: CUDA version not available via JAX API")
        else:
            print("Value: CUDA version not available via JAX API")
    except Exception as e:
        print(f"Value: Error - {e}")
    print()

    print("=" * 80)
    print("Inspection complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
