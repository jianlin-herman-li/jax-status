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
      driverLibHook = ''
        # jaxlib's RPATH points at NixOS-only /run/opengl-driver/lib. Expose this host's driver
        # libs (libcuda + NVML, in /usr/lib/x86_64-linux-gnu) via a curated symlink farm on
        # LD_LIBRARY_PATH; symlinking just these avoids shadowing nix's glibc with the host dir.
        _drv=$(mktemp -d)
        ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$_drv/"
        ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 "$_drv/"
        export LD_LIBRARY_PATH="$_drv''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      '';
    in
    {
      packages.${system}.default = jax-status;

      devShells.${system}.default = pkgs.mkShell {
        packages = (with pkgs; [ python3 fish gh ]) ++ [ jax jax-status ];
        shellHook = driverLibHook;
      };
    };
}
