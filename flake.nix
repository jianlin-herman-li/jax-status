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
        # Get CUDA-enabled JAX for the development shell
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
          cudaPackages = if pkgs.stdenv.isLinux then pkgs.cudaPackages else null;
          makeWrapper = pkgs.makeWrapper;
        };
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
          # Set up CUDA environment for Python to access JAX with GPU support
          shellHook = if pkgs.stdenv.isLinux then
            let
              # Create a Python wrapper that preloads CUDA library
              pythonWrapper = pkgs.writeShellScriptBin "python" ''
                # Preload CUDA driver library before running Python
                export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1:$LD_PRELOAD 2>/dev/null || true
                exec ${python312}/bin/python "$@"
              '';
            in
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
