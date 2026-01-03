{ lib, fetchurl, python312Packages, stdenv }:

let
  version = "9.8.0.87";
  platform = stdenv.hostPlatform.system;
  urlMap = {
    "x86_64-linux" = {
      url = "https://files.pythonhosted.org/packages/77/f0/8236c886a061d203e51247aec2b8e3a8f5350178251ab57237daf2140680/nvidia_cudnn_cu12-9.8.0.87-py3-none-manylinux_2_27_x86_64.whl";
      hash = "sha256-1rAs0OPiSqMdAZOow5/sI5NUNg19gQVe3dtp811TpMg=";
    };
    "aarch64-linux" = {
      url = "https://files.pythonhosted.org/packages/9e/74/57e9771579eada9733610017e782739a370d2a933c4d232ac2db9bda0d8a/nvidia_cudnn_cu12-9.8.0.87-py3-none-manylinux_2_27_aarch64.whl";
      hash = "sha256-uIP66y9vFdunu7Z1bqtqDZzstZ21sPoHV3uc+iTNmfQ=";
    };
  };
  srcInfo = urlMap.${platform} or (throw "Unsupported platform for nvidia-cudnn-cu12 wheel: ${platform}");
  src = fetchurl {
    inherit (srcInfo) url hash;
  };
in
python312Packages.buildPythonPackage {
  pname = "nvidia-cudnn-cu12";
  inherit version src;
  format = "wheel";

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "NVIDIA cuDNN CUDA 12 runtime (wheel)";
    homepage = "https://developer.nvidia.com/cudnn";
    license = licenses.unfreeRedistributable;
    platforms = platforms.linux;
  };
}
