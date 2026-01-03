{ lib, fetchurl, python312Packages, stdenv }:

let
  version = "0.8.1";
  platform = stdenv.hostPlatform.system;
  urlMap = {
    "x86_64-linux" = {
      url = "https://files.pythonhosted.org/packages/eb/4b/3c7e373d81219ee7493c1581c85a926c413ddeb3794cff87a37023a337e4/jaxlib-0.8.1-cp312-cp312-manylinux_2_27_x86_64.whl";
      hash = "sha256-r0kkGJ/FO2kjdxW1bry/xxu5HKFhhBQ9zvDUMMgXPeY=";
    };
    "aarch64-linux" = {
      url = "https://files.pythonhosted.org/packages/7e/73/2aa891de9f5f4c60ba3c63bda97ec4ace50ffb900ff3bf750ce42c514a3b/jaxlib-0.8.1-cp312-cp312-manylinux_2_27_aarch64.whl";
      hash = "sha256-vtHpSujHwWvKRHbY1/WC8NGhAqTmnDqb0gaaDcQidKk=";
    };
    "aarch64-darwin" = {
      url = "https://files.pythonhosted.org/packages/d9/9d/59b36e2f348e599d5812743f263ca54aa03be1a4c9dfc11504d19864b72d/jaxlib-0.8.1-cp312-cp312-macosx_11_0_arm64.whl";
      hash = "sha256-iL3g9TXu6maJ4M1X1At2YNUgaslcfULglWKhCbljpJ8=";
    };
  };
  srcInfo = urlMap.${platform} or (throw "Unsupported platform for jaxlib wheel: ${platform}");
  src = fetchurl {
    inherit (srcInfo) url hash;
  };
in
python312Packages.buildPythonPackage {
  pname = "jaxlib";
  inherit version src;
  format = "wheel";

  propagatedBuildInputs = [
    stdenv.cc.cc.lib
    python312Packages."ml-dtypes"
    python312Packages.numpy
    python312Packages.scipy
  ];

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "XLA library for JAX (source build)";
    homepage = "https://github.com/jax-ml/jax";
    license = licenses.asl20;
    platforms = platforms.linux ++ platforms.darwin;
  };
}
