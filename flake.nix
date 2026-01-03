{
  description = "jax-status";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      allowUnfree = true;
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; config.allowUnfree = allowUnfree; };
        in
        {
          jax-status = pkgs.callPackage ./jax-status.nix { };
          default = self.packages.${system}.jax-status;
        });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; config.allowUnfree = allowUnfree; };
          cudnnWheel =
            if pkgs.stdenv.isLinux && allowUnfree
            then pkgs.callPackage ./nvidia-cudnn-cu12.nix { }
            else null;
          cudaPkgs =
            if pkgs.stdenv.isLinux && allowUnfree
            then [
              pkgs.cudaPackages.cudatoolkit
            ]
            else [ ];
          cudnnWheelLib =
            if cudnnWheel != null
            then "${cudnnWheel}/${pkgs.python312.sitePackages}/nvidia/cudnn/lib"
            else "";
          runtimeLibs =
            if pkgs.stdenv.isLinux
            then cudaPkgs ++ [ pkgs.stdenv.cc.cc.lib ]
            else [ ];
          runtimeLibPath = pkgs.lib.makeLibraryPath runtimeLibs;
          cudaHome =
            if pkgs.stdenv.isLinux && allowUnfree
            then pkgs.cudaPackages.cudatoolkit
            else null;
          driverLibScript = pkgs.writeShellScript "jax-status-driver-libs" ''
            extra_libs=""
            if [ -d /run/opengl-driver/lib ]; then
              extra_libs="/run/opengl-driver/lib"
            fi
            if [ -d /usr/lib/wsl/lib ]; then
              extra_libs="''${extra_libs:+$extra_libs:}/usr/lib/wsl/lib"
            fi
            if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
              driver_dir="''${XDG_RUNTIME_DIR:-/tmp}/jax-status-driver-libs"
              mkdir -p "$driver_dir"
              ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$driver_dir/libcuda.so.1"
              if [ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ]; then
                ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 "$driver_dir/libnvidia-ml.so.1"
              fi
              extra_libs="$driver_dir''${extra_libs:+:$extra_libs}"
              echo "export JAX_DRIVER_LIB_DIR=\"$driver_dir\""
            fi
            if [ -n "$extra_libs" ]; then
              echo "export LD_LIBRARY_PATH=\"$extra_libs:$LD_LIBRARY_PATH\""
            fi
          '';
          shellEnv =
            if pkgs.stdenv.isLinux && allowUnfree
            then {
              CUDA_HOME = cudaHome;
              CUDA_PATH = cudaHome;
              XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaHome}";
              LD_LIBRARY_PATH =
                pkgs.lib.optionalString (cudnnWheelLib != "") "${cudnnWheelLib}:"
                + runtimeLibPath;
            }
            else { };
        in
        {
          default = pkgs.mkShell (shellEnv // {
            packages = [
              pkgs.python312
              (pkgs.callPackage ./jax-status.nix { })
            ] ++ cudaPkgs ++ pkgs.lib.optional (cudnnWheel != null) cudnnWheel;
            shellHook = ''
              ${pkgs.lib.optionalString (pkgs.stdenv.isLinux && allowUnfree) ''
              eval "$(${driverLibScript})"
              ''}
            '';
          });
        });
    };
}
