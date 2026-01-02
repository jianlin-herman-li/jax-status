{ lib, stdenv, python312Packages, config }:

let
  allowUnfree = config.allowUnfree or false;
  allowUnfreePredicate = config.allowUnfreePredicate or (_: false);
  hasJaxlibCuda = lib.hasAttr "jaxlibWithCuda" python312Packages;
  cudaAllowed = allowUnfree || (hasJaxlibCuda && allowUnfreePredicate python312Packages.jaxlibWithCuda);
  jaxlibCuda =
    if hasJaxlibCuda && cudaAllowed
    then python312Packages.jaxlibWithCuda
    else python312Packages.jaxlib;
  jaxlibPkg = if stdenv.isLinux then jaxlibCuda else python312Packages.jaxlib;
  jaxPkg = python312Packages.jax.override { jaxlib = jaxlibPkg; };
in
python312Packages.buildPythonPackage {
  pname = "jax-status";
  version = "0.1.0";
  pyproject = true;
  src = ./.;

  nativeBuildInputs = [
    python312Packages.setuptools
    python312Packages.wheel
  ];

  propagatedBuildInputs = [
    jaxPkg
    jaxlibPkg
  ];

  pythonImportsCheck = [ "jax_status" ];
  doCheck = false;

  meta = {
    description = "Verbose JAX runtime status inspection";
    mainProgram = "jax-status";
  };
}
