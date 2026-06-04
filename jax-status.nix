{
  lib,
  fetchFromGitHub,
  python3Packages,
  jax ? python3Packages.jax,
}:

python3Packages.buildPythonPackage {
  pname = "jax-status";
  version = "0.1.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "jianlin-herman-li";
    repo = "jax-status";
    rev = "7a553df810f086fe3212512c33a93db6fe125f88"; # malphite
    hash = "sha256-91hFAYj3R6AsbQaPXMO9zcZ5AuqmNOWsJgIIDiwwZl8=";
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
