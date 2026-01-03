{ lib, fetchurl, python312Packages, stdenv }:

let
  version = "1.14.1";
  platform = stdenv.hostPlatform.system;
  urlMap = {
    "x86_64-linux" = {
      url = "https://files.pythonhosted.org/packages/8e/ee/8a26858ca517e9c64f84b4c7734b89bda8e63bec85c3d2f432d225bb1886/scipy-1.14.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      hash = "sha256-j56oDy5lvaoLdif7AMvrLa8WPKoBXlm3UWOV/jvR4GY=";
    };
    "aarch64-linux" = {
      url = "https://files.pythonhosted.org/packages/f0/5a/efa92a58dc3a2898705f1dc9dbaf390ca7d4fba26d6ab8cfffb0c72f656f/scipy-1.14.1-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl";
      hash = "sha256-MKyIEsHSqrcTGnm6YpM6Knb1gtXbvGlRkkU9rmetYxA=";
    };
    "x86_64-darwin" = {
      url = "https://files.pythonhosted.org/packages/c0/04/2bdacc8ac6387b15db6faa40295f8bd25eccf33f1f13e68a72dc3c60a99e/scipy-1.14.1-cp312-cp312-macosx_10_13_x86_64.whl";
      hash = "sha256-Yx8Hs3NNNKztAJqvb+39DrNJipflgcOx5fFKBBZKRW0=";
    };
    "aarch64-darwin" = {
      url = "https://files.pythonhosted.org/packages/c8/53/35b4d41f5fd42f5781dbd0dd6c05d35ba8aa75c84ecddc7d44756cd8da2e/scipy-1.14.1-cp312-cp312-macosx_12_0_arm64.whl";
      hash = "sha256-rympNYA8xwerLtd5HEQoimgvnIEHvADw7MxPksCNbgc=";
    };
  };
  srcInfo = urlMap.${platform} or (throw "Unsupported platform for scipy wheel: ${platform}");
  src = fetchurl {
    inherit (srcInfo) url hash;
  };
in
python312Packages.buildPythonPackage {
  pname = "scipy";
  inherit version src;
  format = "wheel";

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "SciPy (wheel)";
    homepage = "https://scipy.org/";
    license = licenses.bsd3;
  };
}
