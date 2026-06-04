{
  description = "jax-status";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaCapabilities = [ "12.0" ]; # NVIDIA RTX PRO 6000 Blackwell (sm_120)
          cudaForwardCompat = false;
        };
      };
      # One python where jax is the CUDA build, so anything depending on it (e.g. optax)
      # inherits the same CUDA jax instead of pulling a conflicting CPU-only jaxlib.
      python = pkgs.python3.override {
        packageOverrides = pyfinal: pyprev: {
          jax = (pyprev.jax.override { cudaSupport = true; }).overridePythonAttrs (_: {
            doCheck = false;
          });
        };
      };
      jax-status = python.pkgs.callPackage ./jax-status.nix {
        python3Packages = python.pkgs;
      };
      # jaxlib's RPATH points at NixOS-only /run/opengl-driver/lib. Expose this host's NVIDIA
      # driver libs (libcuda + NVML, in /usr/lib/x86_64-linux-gnu) as a small symlink farm so
      # LD_LIBRARY_PATH carries just those — pointing it at the whole host dir would shadow
      # nix's glibc and break the toolchain.
      driverLibs = pkgs.linkFarm "nvidia-driver-libs" [
        {
          name = "libcuda.so.1";
          path = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
        }
        {
          name = "libnvidia-ml.so.1";
          path = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1";
        }
      ];
    in
    {
      formatter.${system} = pkgs.nixfmt;

      packages.${system}.default = jax-status;

      devShells.${system}.default = pkgs.mkShell {
        # All Python deps delivered through a single interpreter to keep one coherent jax.
        packages =
          (with pkgs; [
            fish
            gh
          ])
          ++ [
            (python.withPackages (
              ps:
              with ps;
              [
                jax
                matplotlib
                optax
              ]
              ++ [
                jax-status
              ]
            ))
          ];
        shellHook = ''
          export LD_LIBRARY_PATH="${driverLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
