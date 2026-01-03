# All-in-one Nix file for jax-status project
# This file hardcodes everything: package definition, CUDA setup, and development shell
# 
# Usage:
#   - As a flake: pkgs.callPackage ./jax-status-allin1.nix { }
#   - Standalone: nix-shell -E 'let pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.11.tar.gz") { config = { allowUnfree = true; cudaSupport = true; }; }; in (pkgs.callPackage ./jax-status-allin1.nix { }).devShell'
#
# Note: For best results, use with a pinned nixpkgs version (e.g., nixos-24.11)

{ lib
, stdenv
, python3Packages
, python3
, cudaPackages
, runCommand
, writeText
, symlinkJoin
, mkShell
}:

let
  # Use Python 3.12 (python3 should be python312 when called from flake)
  python312 = python3;
  
  # Use CUDA-enabled JAX on Linux
  jaxlibCuda = if stdenv.isLinux && lib.hasAttr "jaxlibWithCuda" python312.pkgs then
    python312.pkgs.jaxlibWithCuda
  else if stdenv.isLinux then
    python312.pkgs.jaxlib.override {
      cudaSupport = true;
      cudaPackages = cudaPackages;
    }
  else
    python312.pkgs.jaxlib;

  # Create CUDA_PATH wrapper that includes symlink to system libcuda.so.1
  # This makes the driver library discoverable via CUDA_PATH/lib/libcuda.so.1
  # Note: The symlink points to the most common system location
  # If the library doesn't exist at that path, the symlink will be broken but harmless
  cudaPathWithDriver = if stdenv.isLinux then
    let
      # Create a directory with symlink to system libcuda.so.1
      # We symlink to the most common location (/usr/lib/x86_64-linux-gnu/libcuda.so.1)
      # The symlink will be resolved at runtime when the library is actually accessed
      driverLibDir = runCommand "cuda-driver-lib" {} ''
        mkdir -p $out/lib
        # Create symlink to most common system location
        # This will be resolved at runtime when accessed
        ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $out/lib/libcuda.so.1
      '';
    in
      # Combine CUDA toolkit with driver library symlink
      symlinkJoin {
        name = "cuda-with-driver";
        paths = [ cudaPackages.cudatoolkit driverLibDir ];
      }
  else
    cudaPackages.cudatoolkit;

  # Hardcoded source files - exact copy-paste from actual files
  pyprojectToml = writeText "pyproject.toml" ''
[project]
name = "jax-status"
version = "0.1.0"
description = "A minimal Python project to inspect JAX runtime status"
requires-python = ">=3.12"
dependencies = [
    "jax[cuda]",
]

[project.scripts]
jax-status = "jax_status.main:main"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["jax_status"]
  '';

  jaxStatusInit = writeText "jax_status/__init__.py" ''
"""JAX status inspection package."""

__version__ = "0.1.0"
  '';

  jaxStatusMain = writeText "jax_status/main.py" ''
#!/usr/bin/env python3
"""Verbose JAX runtime inspection script."""

import sys
import os
import subprocess

# Note: libcuda.so.1 is provided by the system NVIDIA driver, not Nix packages
# CUDA_PATH/lib contains a symlink to the system library, and LD_LIBRARY_PATH
# is set in the Nix shellHook to make it discoverable by the dynamic linker

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
  '';

  # Create source tree from hardcoded files
  sourceTree = runCommand "jax-status-src" {} ''
    mkdir -p $out/jax_status
    cp ${pyprojectToml} $out/pyproject.toml
    cp ${jaxStatusInit} $out/jax_status/__init__.py
    cp ${jaxStatusMain} $out/jax_status/main.py
    chmod +x $out/jax_status/main.py
  '';

  # jax-status package definition (hardcoded, no external file needed)
  jax-status = python312.pkgs.buildPythonPackage rec {
    pname = "jax-status";
    version = "0.1.0";
    format = "pyproject";

    src = sourceTree;

    nativeBuildInputs = with python312.pkgs; [
      setuptools
      wheel
    ];

    propagatedBuildInputs = [
      python312.pkgs.jax
      jaxlibCuda
    ];

    meta = with lib; {
      description = "A minimal Python project to inspect JAX runtime status";
      license = licenses.mit;
      maintainers = [ ];
    };
  };

in
{
  # Package derivation
  package = jax-status;
  
  # Development shell
  devShell = mkShell {
    buildInputs = [
      jax-status
      python312
      python312.pkgs.pip
      # Add JAX with CUDA support to the development shell
      python312.pkgs.jax
      jaxlibCuda
    ] ++ (if stdenv.isLinux then [
      # Add CUDA libraries to the environment
      cudaPackages.cudatoolkit
      cudaPackages.libcublas
      cudaPackages.libcurand
      cudaPackages.libcusolver
      cudaPackages.libcusparse
      cudaPackages.cudnn
    ] else []);
    
    # Set up CUDA environment variables
    shellHook = if stdenv.isLinux then
      ''
        export CUDA_PATH=${cudaPathWithDriver}
        # Add CUDA_PATH/lib to LD_LIBRARY_PATH so libcuda.so.1 symlink is discoverable
        # This makes the Python wrapper unnecessary since the library is in the search path
        export LD_LIBRARY_PATH="${cudaPathWithDriver}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      ''
    else
      "";
  };
}

