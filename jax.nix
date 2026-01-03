{ lib, fetchurl, python312Packages, jaxlibPkg, stdenv, numpyPkg, mlDtypesPkg, scipyPkg, optEinsumPkg }:

let
  version = "0.8.1";
  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/f9/e7/19b8cfc8963b2e10a01a4db7bb27ec5fa39ecd024bc62f8e2d1de5625a9d/jax-0.8.1-py3-none-any.whl";
    hash = "sha256-TL3FVI8wlc3WnTjkM3lQsvwfJQp0CgI00ZDkoxkHdWQ=";
  };
in
python312Packages.buildPythonPackage {
  pname = "jax";
  inherit version src;
  format = "wheel";

  propagatedBuildInputs = [
    stdenv.cc.cc.lib
    jaxlibPkg
    mlDtypesPkg
    numpyPkg
    optEinsumPkg
    scipyPkg
  ];

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "JAX (source build)";
    homepage = "https://github.com/jax-ml/jax";
    license = licenses.asl20;
  };
}
