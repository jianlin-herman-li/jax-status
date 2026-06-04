{ lib
, python3Packages
, jax ? python3Packages.jax
}:

python3Packages.buildPythonPackage {
  pname = "jax-status";
  version = "0.1.0";
  pyproject = true;
  src = ./.;

  nativeBuildInputs = [
    python3Packages.pythonRelaxDepsHook
    python3Packages.setuptools
    python3Packages.wheel
  ];

  pythonRelaxDeps = [
    "jax"
  ];

  propagatedBuildInputs = [
    jax
  ];

  pythonImportsCheck = [ "jax_status" ];
}
