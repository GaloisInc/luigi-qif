{pkgs ? import ./pinned_pkgs.nix }:
with pkgs;
let
  /* For some reason, the version in the Nix repo is marked as linux only. */
  m4ri = stdenv.mkDerivation rec {
    version = "20140914";
    name = "m4ri-${version}";

    src = fetchFromBitbucket {
      owner = "malb";
      repo = "m4ri";
      rev = "release-${version}";
      sha256 = "0xfg6pffbn8r1s0y7bn9b8i55l00d41dkmhrpf7pwk53qa3achd3";
    };

    doCheck = true;

    nativeBuildInputs = [
      autoreconfHook
    ];
  };
  cryptominisat = stdenv.mkDerivation rec{
    name = "cryptominisat";
    version = "5.6.8";
    cmakeFlags = [
      "-DUSE_GAUSS=ON"
      "-DENABLE_PYTHON_INTERFACE=OFF"
      "-DLARGEMEM=1"
    ];
    buildInputs = [
        m4ri
        sqlite
        zlib
        boost
    ];
    nativeBuildInputs = [cmake];
    src = fetchurl {
        url = https://github.com/msoos/cryptominisat/archive/5.6.8.tar.gz;
        sha256 = "0mabfhsixn6jmin6nlahw33fyaagairxy7jgvlmp0yr5qa1d7b9q";
    };
  };
  
  approxmc = stdenv.mkDerivation rec {
    name = "approxmc";
    version = "3.0";
    buildInputs = [
        m4ri
        cryptominisat
        zlib
        boost
    ];
    nativeBuildInputs = [cmake];
    src = fetchurl {
      url = https://github.com/meelgroup/ApproxMC/archive/403a78c662aab107847b8d3185b7eac4ff02e3a1.tar.gz;
      sha256 = "0mwhywbr4h3ylx18j2yi0m5lxkck6y8as12w2r780v7vbx9dq92v";
    };
  };
in {
  py = (python37.withPackages (ps: with ps; [
    z3 click
  ]));
  py2 = python27;
  approxmc = approxmc;
  graphviz = graphviz;
}

