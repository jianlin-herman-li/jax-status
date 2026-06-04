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
          # Expose ONLY the host NVIDIA driver libs (libcuda + NVML) to the nix-built
          # jaxlib, via a curated symlink farm on LD_LIBRARY_PATH. Adding the whole host
          # lib dir would shadow nix's glibc/vdso and break the toolchain.
          driverLibHook = ''
            export PYTHONPATH="$PWD''${PYTHONPATH:+:}$PYTHONPATH"
          '' + nixpkgs.lib.optionalString pkgs.stdenv.isLinux ''
            _drv=$(mktemp -d)
            for _lib in libcuda.so.1 libnvidia-ml.so.1; do
              for _d in /run/opengl-driver/lib /usr/lib/x86_64-linux-gnu /usr/lib64; do
                if [ -e "$_d/$_lib" ]; then ln -sf "$_d/$_lib" "$_drv/$_lib"; break; fi
              done
            done
            export LD_LIBRARY_PATH="$_drv''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        in
        {
          default = pkgs.mkShell {
            packages =
              if pkgs.stdenv.isLinux then
                [ py jaxCuda jaxStatusCuda pkgs.fish pkgs.gh ]
              else
                [ py jaxCpu jaxStatusCpu pkgs.fish pkgs.gh ];
            shellHook = driverLibHook;
          };
          cuda = pkgs.mkShell {
            packages = [ py jaxCuda jaxStatusCuda pkgs.fish pkgs.gh ];
            shellHook = driverLibHook;
          };
        }
      );
    };
}
