{
  description = "jax-status";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs }:
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
      jax =
        (pkgs.python3Packages.jax.override { cudaSupport = true; })
        .overridePythonAttrs (_: { doCheck = false; });
      jax-status = pkgs.callPackage ./jax-status.nix {
        python3Packages = pkgs.python3Packages;
        inherit jax;
      };
      # jaxlib's RPATH points at NixOS-only /run/opengl-driver/lib. Expose this host's NVIDIA
      # driver libs (libcuda + NVML, in /usr/lib/x86_64-linux-gnu) as a small symlink farm so
      # LD_LIBRARY_PATH carries just those — pointing it at the whole host dir would shadow
      # nix's glibc and break the toolchain.
      driverLibs = pkgs.linkFarm "nvidia-driver-libs" [
        { name = "libcuda.so.1"; path = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"; }
        { name = "libnvidia-ml.so.1"; path = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"; }
      ];
    in
    {
      packages.${system}.default = jax-status;

      devShells.${system}.default = pkgs.mkShell {
        packages = (with pkgs; [ python3 fish gh ]) ++ [ jax jax-status ];
        shellHook = ''
          export LD_LIBRARY_PATH="${driverLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
