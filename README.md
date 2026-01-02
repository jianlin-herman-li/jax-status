## Goal

Create a minimal Python project named **jax-status** with the following properties:

- Uses **Python ≥ 3.12**
- Depends on `jax[cuda]`
- Provides a verbose script or CLI that inspects the JAX runtime

## Cost Tracking

Add basic cost and usage tracking to the project.

The system should record the following information for each run:

- The name of the Agent used
- The name of the model used
- Total number of tokens used
- Total cost in USD
- Total wall-clock time spent

After each run, this information must be written to a file named `cost.md` in a clear, human-readable format.

The file should be updated or appended in a consistent way so that multiple runs can be compared over time. 
And add a summary after it succeeds.

## Script Requirements

### Verbose output
The script must be **verbose** and:

- Print **the API being queried** and **the returned value**
- Try **many JAX and backend-related APIs**
- Handle missing or unavailable features gracefully, with clear messages instead of crashes

### Import style rules
- Do **not** use `from ... import ...`
- Use only `import xxx`
- Call APIs using **fully-qualified names** only

### APIs to try (minimum set)
The script should attempt and print results for APIs such as:
- `sys.version`
- `sys.executable`
- `print(subprocess.getoutput("nvidia-smi"))`
- `jax.__version__`
- `jax.lib.__version__`
- `jax.default_backend()`
- `jax.devices()`
- `jax.device_count()`
- `jax.local_devices()`
- `jax.lib.xla_bridge.get_backend().platform`
- `jax.lib.xla_bridge.get_backend().platform_version` when available
- CUDA-related metadata when available, including:
  - CUDA version
  - GPU visibility to JAX
  - Backend and platform name

If CUDA is unavailable, the script must clearly state that.

## GPU Visibility Expectation

On Linux systems with NVIDIA GPUs:

- If `nvidia-smi` reports available GPUs
- And the user runs `nix develop -c jax-status`
- Then **JAX must report GPU devices as available**

Failure to see GPUs in JAX under these conditions must be clearly highlighted in the output.

## Project Structure

The repository should contain:

- `pyproject.toml`  
  A minimal Python project definition for **jax-status**.

- `jax-status/`  
  Python package implementing the verbose JAX status inspection logic.

- `flake.nix`  
  Nix flake entry point.

- `jax-status.nix`  
  Nix derivation or module defining the development environment.

- `jax-status-singleton.nix`
  Should be the same as `jax-status.nix` except that the Python project (`pyproject.toml` and `jax-status/`) is hard-coded in it.
  
## Nix Requirements

- Use **NixOS 24.11**
- `nix develop` must work on macOS and Ubuntu Linux
- The development shell should:
  - Provide Python 3.12
  - Install JAX with CUDA support on Linux when possible
  - Gracefully fall back to CPU-only JAX on macOS

## Expected Workflow

1. Implement the minimal `pyproject.toml`.
2. Implement the verbose JAX status script following the import and API rules above.
3. Write `flake.nix` and `jax-status.nix`.
4. Run `nix develop -c jax-status`.
5. Iteratively fix errors until `nix develop -c jax-status` succeeds on both platforms.
6. Run the script and verify that:
   - Each queried API name is printed
   - Each corresponding value is printed
   - GPU visibility matches `nvidia-smi` on Linux systems
   - Missing or unsupported features are clearly reported
