{
  description = "JAX status inspection tool";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            # Enable CUDA support on Linux
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        # Use Python 3.12
        python312 = pkgs.python312;
        # Use CUDA-enabled JAX on Linux
        jaxlibCuda = if pkgs.stdenv.isLinux && pkgs.lib.hasAttr "jaxlibWithCuda" python312.pkgs then
          python312.pkgs.jaxlibWithCuda
        else if pkgs.stdenv.isLinux then
          python312.pkgs.jaxlib.override {
            cudaSupport = true;
            cudaPackages = pkgs.cudaPackages;
          }
        else
          python312.pkgs.jaxlib;
        jax-status = pkgs.callPackage ./jax-status.nix {
          python3 = python312;
          python3Packages = python312.pkgs;
          jaxlib = jaxlibCuda;
        };

        # Create Python wrapper that preloads CUDA driver library
        # Note: libcuda.so.1 is provided by system NVIDIA driver, not Nix packages
        pythonWrapper = if pkgs.stdenv.isLinux then
          pkgs.writeShellScriptBin "python" ''
            # Try to preload system CUDA driver library if available
            if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
              export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1:$LD_PRELOAD
            elif [ -f /usr/lib/libcuda.so.1 ]; then
              export LD_PRELOAD=/usr/lib/libcuda.so.1:$LD_PRELOAD
            fi
            exec ${python312}/bin/python "$@"
          ''
        else null;
      in
      {
        packages.default = jax-status;
        packages.jax-status = jax-status;

        devShells.default = pkgs.mkShell {
          buildInputs = [
            jax-status
            python312
            python312.pkgs.pip
            # Add JAX with CUDA support to the development shell
            python312.pkgs.jax
            jaxlibCuda
          ] ++ (if pkgs.stdenv.isLinux then [
            # Add CUDA libraries to the environment
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.libcublas
            pkgs.cudaPackages.libcurand
            pkgs.cudaPackages.libcusolver
            pkgs.cudaPackages.libcusparse
            pkgs.cudaPackages.cudnn
          ] else []);
          # Set up CUDA environment variables
          shellHook = if pkgs.stdenv.isLinux then
            ''
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              # Use Python wrapper that preloads CUDA
              export PATH="${pythonWrapper}/bin:$PATH"
            ''
          else
            "";
        };
      });
}
