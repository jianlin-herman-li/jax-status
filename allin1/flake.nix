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
        
        # Call the all-in-one Nix file as a derivation
        # Override python3 to use python312
        allin1 = pkgs.callPackage ./jax-status-allin1.nix {
          python3 = pkgs.python312;
        };
      in
      {
        packages.default = allin1.package;
        packages.jax-status = allin1.package;

        devShells.default = allin1.devShell;
      });
}

