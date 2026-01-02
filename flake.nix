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
          cudaPkgs =
            if pkgs.stdenv.isLinux && allowUnfree
            then [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
            ]
            else [ ];
          cudaLibPath = pkgs.lib.makeLibraryPath cudaPkgs;
          cudaHome =
            if pkgs.stdenv.isLinux && allowUnfree
            then pkgs.cudaPackages.cudatoolkit
            else null;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python312
              (pkgs.callPackage ./jax-status.nix { })
            ] ++ cudaPkgs;
            shellHook = ''
              ${pkgs.lib.optionalString (pkgs.stdenv.isLinux && allowUnfree) ''
              export CUDA_HOME=${cudaHome}
              export CUDA_PATH=${cudaHome}
              export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cudaHome}"
              export LD_LIBRARY_PATH="${cudaLibPath}:$LD_LIBRARY_PATH"
              if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
              fi
              if [ -d /usr/lib/wsl/lib ]; then
                export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
              fi
              if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
                driver_dir="$PWD/.nvidia-driver-libs"
                mkdir -p "$driver_dir"
                ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$driver_dir/libcuda.so.1"
                if [ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ]; then
                  ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 "$driver_dir/libnvidia-ml.so.1"
                fi
                export LD_LIBRARY_PATH="$driver_dir:$LD_LIBRARY_PATH"
              fi
              ''}
            '';
          };
        });
    };
}
