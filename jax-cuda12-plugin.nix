{ lib, fetchurl, python312Packages, stdenv }:

let
  version = "0.8.1";
  platform = stdenv.hostPlatform.system;
  urlMap = {
    "x86_64-linux" = {
      url = "https://files.pythonhosted.org/packages/20/60/1dde369dd70b349ff388cd699d69c7d49ff3494af30b5b774037cc4d45e6/jax_cuda12_plugin-0.8.1-cp312-cp312-manylinux_2_27_x86_64.whl";
      hash = "sha256-tgvwu9okzsb6cRcL1pthM1nwGjdtjgn+NL9n7MmjFk8=";
    };
    "aarch64-linux" = {
      url = "https://files.pythonhosted.org/packages/89/cb/8119088cab8d798ca4a18d1ed143be3d90057c2fa2e8dbaf3bfff779014d/jax_cuda12_plugin-0.8.1-cp312-cp312-manylinux_2_27_aarch64.whl";
      hash = "sha256-OFAB9W+FKVnwYa4VrRV8OcxEccjR0lRN/D+AVoSsIhM=";
    };
  };
  srcInfo = urlMap.${platform} or (throw "Unsupported platform for jax-cuda12-plugin wheel: ${platform}");
  src = fetchurl {
    inherit (srcInfo) url hash;
  };
in
python312Packages.buildPythonPackage {
  pname = "jax-cuda12-plugin";
  inherit version src;
  format = "wheel";

  dontCheckRuntimeDeps = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "JAX CUDA 12 plugin (source build)";
    homepage = "https://github.com/jax-ml/jax";
    license = licenses.asl20;
    platforms = platforms.linux;
  };
}
