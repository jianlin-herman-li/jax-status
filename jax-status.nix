{ callPackage, lib, stdenv, python312Packages }:

let
  numpyPkg = python312Packages.numpy_2;
  mlDtypesPkg = callPackage ./ml-dtypes.nix { };
  scipyPkg = callPackage ./scipy.nix { };
  optEinsumPkg = callPackage ./opt-einsum.nix { };
  jaxlibPkg = callPackage ./jaxlib.nix { inherit numpyPkg mlDtypesPkg scipyPkg; };
  jaxPkg = callPackage ./jax.nix { inherit jaxlibPkg numpyPkg mlDtypesPkg scipyPkg optEinsumPkg; };
  cudaPluginPkg =
    if stdenv.isLinux
    then callPackage ./jax-cuda12-plugin.nix { }
    else null;
  cudaPjrtPkg =
    if stdenv.isLinux
    then callPackage ./jax-cuda12-pjrt.nix { }
    else null;
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

  propagatedBuildInputs =
    [
      jaxPkg
      jaxlibPkg
      numpyPkg
      scipyPkg
    ]
    ++ lib.optional (cudaPluginPkg != null) cudaPluginPkg
    ++ lib.optional (cudaPjrtPkg != null) cudaPjrtPkg;

  pythonImportsCheck = [ "jax_status" ];
  doCheck = false;

  meta = {
    description = "Verbose JAX runtime status inspection";
    mainProgram = "jax-status";
  };
}
