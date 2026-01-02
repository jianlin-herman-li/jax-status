{ lib
, python312
, stdenv
, cudaPackages ? null
, makeWrapper
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

  jax-status-pkg = python312.pkgs.buildPythonPackage rec {
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
  };
in

# Wrap the binary to set CUDA_PATH on Linux when cudaPackages is available
if stdenv.isLinux && cudaPackages != null then
  stdenv.mkDerivation {
    pname = "jax-status";
    version = jax-status-pkg.version;
    inherit (jax-status-pkg) meta;

    buildInputs = [ makeWrapper ];

    buildCommand = ''
      mkdir -p $out/bin
      makeWrapper ${jax-status-pkg}/bin/jax-status $out/bin/jax-status \
        --set CUDA_PATH ${cudaPackages.cudatoolkit}
    '';
  }
else
  jax-status-pkg
