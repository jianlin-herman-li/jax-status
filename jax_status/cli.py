import os
import platform
import subprocess
import sys
import time


def _print_section(title):
    print("\n" + "=" * 4 + " " + title + " " + "=" * 4)


def _query(name, func):
    print("API: " + name)
    try:
        value = func()
        print("VALUE: " + str(value))
        return True, value
    except Exception as exc:
        print("ERROR: " + type(exc).__name__ + ": " + str(exc))
        return False, None


def _env_first(keys, default="unknown"):
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return default


def _append_cost(start_wall, end_wall, wall_time, status):
    agent_name = _env_first(["CODEX_AGENT", "AGENT_NAME", "OPENAI_AGENT"], "unknown")
    model_name = _env_first(["CODEX_MODEL", "OPENAI_MODEL", "MODEL_NAME"], "unknown")
    total_tokens = _env_first(
        [
            "CODEX_TOTAL_TOKENS",
            "OPENAI_TOTAL_TOKENS",
            "OPENAI_USAGE_TOTAL_TOKENS",
            "TOTAL_TOKENS",
        ],
        "unknown",
    )
    total_cost = _env_first(
        [
            "CODEX_TOTAL_COST_USD",
            "OPENAI_TOTAL_COST_USD",
            "OPENAI_USAGE_TOTAL_COST",
            "TOTAL_COST_USD",
        ],
        "unknown",
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_wall))

    lines = [
        "## Run " + timestamp,
        "- Agent: " + agent_name,
        "- Model: " + model_name,
        "- Total tokens: " + str(total_tokens),
        "- Total cost (USD): " + str(total_cost),
        "- Wall time (s): " + "{:.3f}".format(wall_time),
        "- Status: " + status,
        "- Summary: Agent="
        + agent_name
        + ", Model="
        + model_name
        + ", Tokens="
        + str(total_tokens)
        + ", CostUSD="
        + str(total_cost)
        + ", WallTimeS="
        + "{:.3f}".format(wall_time),
        "",
    ]

    with open("cost.md", "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _nvidia_smi_has_gpus(output):
    if not output:
        return False
    if "NVIDIA-SMI" not in output:
        return False
    if "GPU" in output:
        return True
    return False


def main():
    start_wall = time.time()
    start_perf = time.perf_counter()
    status = "ok"

    _print_section("System")
    _query("sys.version", lambda: sys.version)
    _query("sys.executable", lambda: sys.executable)
    _query("platform.system", lambda: platform.system())
    _query("platform.machine", lambda: platform.machine())
    _query("os.environ.get('CUDA_VISIBLE_DEVICES')", lambda: os.environ.get("CUDA_VISIBLE_DEVICES"))
    _query("os.environ.get('JAX_PLATFORM_NAME')", lambda: os.environ.get("JAX_PLATFORM_NAME"))
    _query("os.environ.get('LD_LIBRARY_PATH')", lambda: os.environ.get("LD_LIBRARY_PATH"))
    _query("os.environ.get('JAX_DRIVER_LIB_DIR')", lambda: os.environ.get("JAX_DRIVER_LIB_DIR"))
    _query("os.path.exists('/dev/nvidia0')", lambda: os.path.exists("/dev/nvidia0"))
    _query("os.path.exists('/dev/nvidiactl')", lambda: os.path.exists("/dev/nvidiactl"))
    _query(
        "os.path.exists(os.environ.get('JAX_DRIVER_LIB_DIR', ''))",
        lambda: os.path.exists(os.environ.get("JAX_DRIVER_LIB_DIR", "")),
    )

    _print_section("NVIDIA")
    ok_smi, smi_output = _query(
        "subprocess.getoutput('nvidia-smi')",
        lambda: subprocess.getoutput("nvidia-smi"),
    )
    _query(
        "subprocess.getoutput('ldconfig -p | grep -i libcuda')",
        lambda: subprocess.getoutput("ldconfig -p | grep -i libcuda"),
    )
    _query(
        "subprocess.getoutput('ls -l $JAX_DRIVER_LIB_DIR 2>/dev/null')",
        lambda: subprocess.getoutput("ls -l ${JAX_DRIVER_LIB_DIR:-/no-jax-driver-lib-dir} 2>/dev/null"),
    )
    smi_has_gpus = ok_smi and _nvidia_smi_has_gpus(smi_output)
    if ok_smi and not smi_has_gpus:
        print("NOTE: nvidia-smi did not report GPUs or is unavailable.")

    _print_section("JAX")
    try:
        import jax
    except Exception as exc:
        print("ERROR: Failed to import jax: " + type(exc).__name__ + ": " + str(exc))
        status = "error"
        end_perf = time.perf_counter()
        end_wall = time.time()
        _append_cost(start_wall, end_wall, end_perf - start_perf, status)
        return 1

    _query("jax.__version__", lambda: jax.__version__)
    _query("jax.lib.__version__", lambda: jax.lib.__version__)
    _query("jax.default_backend()", lambda: jax.default_backend())
    _query("jax.devices()", lambda: jax.devices())
    _query("jax.device_count()", lambda: jax.device_count())
    _query("jax.local_devices()", lambda: jax.local_devices())
    _query("jax.config.read('jax_platform_name')", lambda: jax.config.read("jax_platform_name"))

    def _jax_platforms():
        if hasattr(jax.config, "jax_platforms"):
            return jax.config.jax_platforms
        return "unavailable (no jax_platforms attribute)"

    _query("jax.config.jax_platforms", _jax_platforms)
    _query(
        "jax.lib.xla_bridge.get_backend().platform",
        lambda: jax.lib.xla_bridge.get_backend().platform,
    )

    def _platform_version():
        backend = jax.lib.xla_bridge.get_backend()
        if hasattr(backend, "platform_version"):
            return backend.platform_version
        return "unavailable (no platform_version attribute)"

    _query("jax.lib.xla_bridge.get_backend().platform_version", _platform_version)

    def _cuda_version_hint():
        backend = jax.lib.xla_bridge.get_backend()
        platform_version = getattr(backend, "platform_version", "")
        if not platform_version:
            return "unavailable (no platform_version)"
        if "CUDA" in platform_version or "cuda" in platform_version:
            return platform_version
        return "platform_version has no CUDA marker: " + str(platform_version)

    _query("CUDA version hint", _cuda_version_hint)

    def _jax_gpu_devices():
        return [device for device in jax.devices() if device.platform in ("gpu", "cuda")]

    ok_gpu_list, gpu_devices = _query("JAX GPU devices", _jax_gpu_devices)
    if ok_gpu_list:
        if smi_has_gpus and len(gpu_devices) == 0:
            print("!!! GPU MISMATCH: nvidia-smi reports GPUs but JAX sees none.")
        elif smi_has_gpus:
            print("GPU visibility check: JAX reports GPU devices and nvidia-smi reports GPUs.")
        else:
            print("GPU visibility check: nvidia-smi unavailable or no GPUs reported.")

    end_perf = time.perf_counter()
    end_wall = time.time()
    _append_cost(start_wall, end_wall, end_perf - start_perf, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
