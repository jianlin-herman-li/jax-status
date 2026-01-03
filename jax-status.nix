{ callPackage, lib, stdenv, python312Packages }:

let
  jaxlibPkg = callPackage ./jaxlib.nix { };
  jaxPkg = callPackage ./jax.nix { inherit jaxlibPkg; };
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
