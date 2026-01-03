## Goal

Create a minimal Python project named **jax-status** with the following properties:

- Uses **Python ≥ 3.12**
- Depends on `jax[cuda]`
- Provides a verbose script or CLI that inspects the JAX runtime
- Implementation should be simple and minimal; there is no need to be abstract.

## Conversation Logging

Persist the full conversation between the user and the system.

After each interaction:

- Save the exchanged messages between the user and the assistant
- Preserve message order and roles
- Use a clear, readable Markdown format

The conversation must be written to a file named `conversation.md`.

If the file already exists, append new interactions in chronological order so the full history is retained.

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
  
### Import style rules
- Do **not** use `from ... import ...`
- Use only `import xxx`
- Call APIs using **fully-qualified names** only

### APIs to try (minimum set)
The script should attempt and print results for APIs when avaliable such as:
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
- `jax.lib.xla_bridge.get_backend().platform_version`
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

- `jax_status/`  
  Python package implementing the verbose JAX status inspection logic.

- `flake.nix`  
  Nix flake entry point.
  Use nixos 25.05 or 25.11

- `jax-status.nix`  
  A Nix file that wraps this pyproject into a Nix derivation.
  Use `buildPythonPackage`.
  
Nix files should be short and simple if possible.
  
## Nix Requirements

- Use **NixOS 25.05 or 25.11**
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

## Note

Make `jax-status` a true nix derivation via `buildPythonPackage`, don't use `pip install` in flake.nix.

Automatically run `nix develop -c jax-status`, and fix errors until it succeeds.
