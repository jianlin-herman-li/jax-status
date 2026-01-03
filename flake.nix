{
  description = "jax-status";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-cuda.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs, nixpkgs-cuda }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      packages = forAllSystems (system:
        let
          pkgsCuda = import nixpkgs-cuda {
            inherit system;
            config = {
              allowUnfree = true;
              cudaCapabilities = [ "7.5" ];
              cudaForwardCompat = false;
            };
          };
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaCapabilities = [ "7.5" ];
              cudaForwardCompat = false;
            };
          };
          pyPkgs = if pkgs.stdenv.isLinux then pkgsCuda.python312Packages else pkgs.python312Packages;
          jaxCpu = pyPkgs.jax.overridePythonAttrs (_: { doCheck = false; });
          jaxCuda =
            if pkgs.stdenv.isLinux then
              (pyPkgs.jax.override { cudaSupport = true; }).overridePythonAttrs (_: { doCheck = false; })
            else
              jaxCpu;
          jaxStatusCpu = pkgs.callPackage ./jax-status.nix { python312Packages = pyPkgs; jax = jaxCpu; };
          jaxStatusCuda = pkgs.callPackage ./jax-status.nix { python312Packages = pyPkgs; jax = jaxCuda; };
        in
        {
          default = if pkgs.stdenv.isLinux then jaxStatusCuda else jaxStatusCpu;
          cuda = jaxStatusCuda;
        }
      );

      devShells = forAllSystems (system:
        let
          pkgsCuda = import nixpkgs-cuda {
            inherit system;
            config = {
              allowUnfree = true;
              cudaCapabilities = [ "7.5" ];
              cudaForwardCompat = false;
            };
          };
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaCapabilities = [ "7.5" ];
              cudaForwardCompat = false;
            };
          };
          py = if pkgs.stdenv.isLinux then pkgsCuda.python312 else pkgs.python312;
          pyPkgs = if pkgs.stdenv.isLinux then pkgsCuda.python312Packages else pkgs.python312Packages;
          jaxCpu = pyPkgs.jax.overridePythonAttrs (_: { doCheck = false; });
          jaxCuda =
            if pkgs.stdenv.isLinux then
              (pyPkgs.jax.override { cudaSupport = true; }).overridePythonAttrs (_: { doCheck = false; })
            else
              jaxCpu;
          jaxStatusCpu = pkgs.callPackage ./jax-status.nix { python312Packages = pyPkgs; jax = jaxCpu; };
          jaxStatusCuda = pkgs.callPackage ./jax-status.nix { python312Packages = pyPkgs; jax = jaxCuda; };
        in
        {
          default = pkgs.mkShell {
            packages =
              if pkgs.stdenv.isLinux then
                [ py jaxCuda jaxStatusCuda ]
              else
                [ py jaxCpu jaxStatusCpu ];
            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:}$PYTHONPATH"
            '';
          };
          cuda = pkgs.mkShell {
            packages = [ py jaxCuda jaxStatusCuda ];
            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:}$PYTHONPATH"
            '';
          };
        }
      );
    };
}
