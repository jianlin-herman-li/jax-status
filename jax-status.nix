{ lib
, fetchFromGitHub
, python3Packages
, jax ? python3Packages.jax
}:

python3Packages.buildPythonPackage {
  pname = "jax-status";
  version = "0.1.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "jianlin-herman-li";
    repo = "jax-status";
    rev = "b72f780183889df2aa40754052b444191fce4a55"; # malphite
    hash = "sha256-jBexDZoyCVoFx/kP8Q9fgMfoCNA+AIoDv5pf9VNUg3A=";
  };

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
