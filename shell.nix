{pkgs ? import /mnt/x/Dev/nixpkgs {}}:
with pkgs;
  mkShell {
    buildInputs = [
      python312Packages.python
      python312Packages.venvShellHook
      python312Packages.mediapipe
      (python312Packages.opencv4.override {
        enableGtk2 = true;
      })
      python312Packages.numpy

      libGL
      libglvnd
      cmake
    ];
    packages = [pkgs.poetry];
    venvDir = "./.venv";
    postVenvCreation = ''
      unset SOURCE_DATE_EPOCH
      poetry env use .venv/bin/python
      poetry install
    '';
    postShellHook = ''
      unset SOURCE_DATE_EPOCH
      export LD_LIBRARY_PATH=${lib.makeLibraryPath [stdenv.cc.cc]}
      poetry env info
    '';
  }
