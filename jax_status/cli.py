import os
import platform
import subprocess
import sys
import time

def _ensure_system_cuda_driver():
    if platform.system().lower() != "linux":
        return
    candidate_libs = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
    ]
    for lib_path in candidate_libs:
        if os.path.exists(lib_path):
            try:
                import ctypes
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                return
            return

def _print_value(api_name, value):
    print(f"API: {api_name}")
    print(f"VALUE: {value}")

def _print_error(api_name, exc):
    print(f"API: {api_name}")
    print(f"VALUE: ERROR: {type(exc).__name__}: {exc}")

def _try_call(api_name, func):
    try:
        value = func()
    except Exception as exc:
        _print_error(api_name, exc)
        return None
    _print_value(api_name, value)
    return value

def _nvidia_smi_output():
    return subprocess.getoutput("nvidia-smi")

def _has_nvidia_gpu(nvidia_smi_text):
    if "NVIDIA-SMI" not in nvidia_smi_text:
        return False
    if "No devices were found" in nvidia_smi_text:
        return False
    if "not found" in nvidia_smi_text:
        return False
    return True

def _report_gpu_visibility(nvidia_smi_text, jax_devices):
    has_nvidia = _has_nvidia_gpu(nvidia_smi_text)
    if not has_nvidia:
        _print_value("cuda.visibility", "CUDA not available via nvidia-smi")
        return

    jax_gpu_devices = []
    if jax_devices is not None:
        for device in jax_devices:
            if getattr(device, "platform", "") in ["gpu", "cuda"]:
                jax_gpu_devices.append(device)

    if not jax_gpu_devices:
        _print_value(
            "cuda.visibility",
            "WARNING: nvidia-smi reports GPUs but JAX does not see them",
        )
    else:
        _print_value(
            "cuda.visibility",
            f"JAX sees GPU devices: {len(jax_gpu_devices)}",
        )

def main():
    start_time = time.time()
    _print_value("sys.version", sys.version)
    _print_value("sys.executable", sys.executable)
    _print_value("platform.system", platform.system())
    _print_value("platform.platform", platform.platform())

    nvidia_smi_text = _try_call("subprocess.getoutput(\"nvidia-smi\")", _nvidia_smi_output)

    _ensure_system_cuda_driver()

    try:
        import jax
    except Exception as exc:
        _print_error("import jax", exc)
        return 1

    _print_value("jax.__version__", jax.__version__)
    _print_value("jax.lib.__version__", jax.lib.__version__)

    _try_call("jax.default_backend()", jax.default_backend)
    jax_devices = _try_call("jax.devices()", jax.devices)
    _try_call("jax.device_count()", jax.device_count)
    _try_call("jax.local_devices()", jax.local_devices)

    _try_call(
        "jax.lib.xla_bridge.get_backend().platform",
        lambda: jax.lib.xla_bridge.get_backend().platform,
    )
    _try_call(
        "jax.lib.xla_bridge.get_backend().platform_version",
        lambda: jax.lib.xla_bridge.get_backend().platform_version,
    )

    if platform.system().lower() == "linux":
        if nvidia_smi_text is None:
            _print_value("cuda.visibility", "nvidia-smi output not available")
        else:
            _report_gpu_visibility(nvidia_smi_text, jax_devices)
    else:
        _print_value("cuda.visibility", "CUDA not expected on this platform")

    _print_value("runtime.wall_clock_seconds", round(time.time() - start_time, 3))
    _print_value("runtime.cwd", os.getcwd())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
