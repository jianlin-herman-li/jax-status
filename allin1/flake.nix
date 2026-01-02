{
  description = "JAX status inspection tool (all-in-one version)";

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
        
        # Import the all-in-one Nix file
        # Path is relative to the flake.nix location (allin1 directory)
        allin1 = import ./jax-status-allin1.nix {
          inherit pkgs;
          lib = pkgs.lib;
        };
      in
      {
        packages.default = allin1.packages.jax-status;
        packages.jax-status = allin1.packages.jax-status;

        devShells.default = allin1.devShell;
      });
}

