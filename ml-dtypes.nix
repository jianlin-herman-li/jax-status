{ lib, fetchurl, python312Packages, stdenv }:

let
  version = "0.5.0";
  platform = stdenv.hostPlatform.system;
  urlMap = {
    "x86_64-linux" = {
      url = "https://files.pythonhosted.org/packages/6f/d3/1321715a95e856d4ef4fba24e4351cf5e4c89d459ad132a8cba5fe257d72/ml_dtypes-0.5.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      hash = "sha256-o43432EZSuquGrdXkHV3m0rTLNHP/QEsKL4if6fypwo=";
    };
    "aarch64-linux" = {
      url = "https://files.pythonhosted.org/packages/31/75/bf571247bb3dbea73aa33ccae57ce322b9688003cfee2f68d303ab7b987b/ml_dtypes-0.5.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl";
      hash = "sha256-qYi6xlcmMOHpwu3ZsSd7Tu/RyGIJ5SsNBht3WsM5Av8=";
    };
    "x86_64-darwin" = {
      url = "https://files.pythonhosted.org/packages/1c/b7/a067839f6e435785f34b09d96938dccb3a5d9502037de243cb84a2eb3f23/ml_dtypes-0.5.0-cp312-cp312-macosx_10_9_universal2.whl";
      hash = "sha256-1LGnCj5SGXkNa1W5UHYG/E4CkR0Ul9FsGN1yHrfv59A=";
    };
    "aarch64-darwin" = {
      url = "https://files.pythonhosted.org/packages/1c/b7/a067839f6e435785f34b09d96938dccb3a5d9502037de243cb84a2eb3f23/ml_dtypes-0.5.0-cp312-cp312-macosx_10_9_universal2.whl";
      hash = "sha256-1LGnCj5SGXkNa1W5UHYG/E4CkR0Ul9FsGN1yHrfv59A=";
    };
  };
  srcInfo = urlMap.${platform} or (throw "Unsupported platform for ml-dtypes wheel: ${platform}");
  src = fetchurl {
    inherit (srcInfo) url hash;
  };
in
python312Packages.buildPythonPackage {
  pname = "ml-dtypes";
  inherit version src;
  format = "wheel";

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "ml-dtypes (wheel)";
    homepage = "https://github.com/jax-ml/ml_dtypes";
    license = licenses.asl20;
  };
}
