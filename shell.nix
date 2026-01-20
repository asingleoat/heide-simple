# shell.nix
let
  pkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/24.05.tar.gz";
    # sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
  }) {};

  python = pkgs.python3;

  py = python.withPackages (ps: with ps; [
    # core science
    numpy
    scipy
    pandas
    sympy

    # image processing
    pillow
    imageio
    scikit-image
    # rawpy
    opencv4

    # general utilities
    requests
    tqdm
    pyyaml

    # dev tooling
    pytest
    pytest-xdist
    black
    # ruff
  ]);
in
pkgs.mkShell {
  packages = [
    py
    pkgs.git
    pkgs.bashInteractive
    pkgs.coreutils
    pkgs.pkg-config
    pkgs.gcc
  ];

  shellHook = ''
    export PYTHONNOUSERSITE=1
    echo "Python: $(python --version)"
  '';
}
