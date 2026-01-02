{ lib
, python312
, stdenv
, cudaPackages ? null
}:

let
  # Use CUDA-enabled JAX on Linux
  jaxlibCuda = if stdenv.isLinux && lib.hasAttr "jaxlibWithCuda" python312.pkgs then
    python312.pkgs.jaxlibWithCuda
  else if stdenv.isLinux && cudaPackages != null then
    python312.pkgs.jaxlib.override {
      cudaSupport = true;
      cudaPackages = cudaPackages;
    }
  else
    python312.pkgs.jaxlib;
in

python312.pkgs.buildPythonPackage rec {
  pname = "jax-status";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = with python312.pkgs; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = [
    python312.pkgs.jax
    jaxlibCuda
  ];

  meta = with lib; {
    description = "A minimal Python project to inspect JAX runtime status";
    license = licenses.mit;
    maintainers = [ ];
  };
}
