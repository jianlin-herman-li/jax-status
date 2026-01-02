{ lib
, python3
, python3Packages
, stdenv
, cudaPackages ? null
}:

let
  # Use CUDA-enabled JAX on Linux
  jaxlibCuda = if stdenv.isLinux && lib.hasAttr "jaxlibWithCuda" python3Packages then
    python3Packages.jaxlibWithCuda
  else if stdenv.isLinux && cudaPackages != null then
    python3Packages.jaxlib.override {
      cudaSupport = true;
      cudaPackages = cudaPackages;
    }
  else
    python3Packages.jaxlib;
in

python3Packages.buildPythonPackage rec {
  pname = "jax-status";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = with python3Packages; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = [
    python3Packages.jax
    jaxlibCuda
  ];

  meta = with lib; {
    description = "A minimal Python project to inspect JAX runtime status";
    license = licenses.mit;
    maintainers = [ ];
  };
}
