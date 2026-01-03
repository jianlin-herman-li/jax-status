{ lib, fetchurl, python312Packages }:

let
  version = "3.4.0";
  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/23/cd/066e86230ae37ed0be70aae89aabf03ca8d9f39c8aea0dec8029455b5540/opt_einsum-3.4.0-py3-none-any.whl";
    hash = "sha256-abuSRp+GoVZRlezkrAMjlD6DR3FxuR0kw1r+AoqQ180=";
  };
in
python312Packages.buildPythonPackage {
  pname = "opt-einsum";
  inherit version src;
  format = "wheel";

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "Optimizing einsum operations (wheel)";
    homepage = "https://github.com/dgasmith/opt_einsum";
    license = licenses.mit;
  };
}
