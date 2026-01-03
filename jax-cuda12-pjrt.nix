{ lib, fetchurl, python312Packages, stdenv }:

let
  version = "0.8.1";
  platform = stdenv.hostPlatform.system;
  urlMap = {
    "x86_64-linux" = {
      url = "https://files.pythonhosted.org/packages/c1/85/c59752caca94e72861f7a6a42f37485df706e60ec4bb27090081899001d4/jax_cuda12_pjrt-0.8.1-py3-none-manylinux_2_27_x86_64.whl";
      hash = "sha256-RStw7hDLmsXX38pV/7zbibbIvGunCkWvfEkNHc6pjrc=";
    };
    "aarch64-linux" = {
      url = "https://files.pythonhosted.org/packages/a3/e4/53e6f7bb36bfe0b9223deaffc083c5c3e1ac9110837c1ef1139c9669b3a8/jax_cuda12_pjrt-0.8.1-py3-none-manylinux_2_27_aarch64.whl";
      hash = "sha256-pjHQaJkDNUr9ez0uxZW32gamIwp22gD/lUj1QrIbYlA=";
    };
  };
  srcInfo = urlMap.${platform} or (throw "Unsupported platform for jax-cuda12-pjrt wheel: ${platform}");
  src = fetchurl {
    inherit (srcInfo) url hash;
  };
in
python312Packages.buildPythonPackage {
  pname = "jax-cuda12-pjrt";
  inherit version src;
  format = "wheel";

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "JAX CUDA 12 PJRT plugin (wheel)";
    homepage = "https://github.com/jax-ml/jax";
    license = licenses.asl20;
    platforms = platforms.linux;
  };
}
