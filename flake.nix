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
        jax-status = pkgs.callPackage ./jax-status.nix {
          cudaPackages = if pkgs.stdenv.isLinux then pkgs.cudaPackages else null;
        };
      in
      {
        packages.default = jax-status;
        packages.jax-status = jax-status;

        devShells.default = pkgs.mkShell {
          buildInputs = [
            jax-status
            pkgs.python312
            pkgs.python312.pkgs.pip
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
            ''
          else
            "";
        };
      });
}
