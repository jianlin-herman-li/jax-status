{ lib
, python3
, python3Packages
, jaxlib ? python3Packages.jaxlib
}:

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
    jaxlib
  ];

  meta = with lib; {
    description = "A minimal Python project to inspect JAX runtime status";
    license = licenses.mit;
    maintainers = [ ];
  };
}
