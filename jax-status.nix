{ lib
, python312Packages
, jax ? python312Packages.jax
}:

python312Packages.buildPythonPackage {
  pname = "jax-status";
  version = "0.1.0";
  pyproject = true;
  src = ./.;

  nativeBuildInputs = [
    python312Packages.pythonRelaxDepsHook
    python312Packages.setuptools
    python312Packages.wheel
  ];

  pythonRelaxDeps = [
    "jax"
  ];

  propagatedBuildInputs = [
    jax
  ];

  pythonImportsCheck = [ "jax_status" ];
}
