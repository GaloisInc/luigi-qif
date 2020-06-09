let 
  pinnedPkgs = builtins.fetchTarball {
    name = "nixpkgs";
    url = https://github.com/NixOS/nixpkgs-channels/archive/07b42ccf2de451342982b550657636d891c4ba35.tar.gz;
    sha256 = "1a7ga18pwq0y4p9r787622ry8gssw1p6wsr5l7dl8pqnj1hbbzwh";
  };
in import pinnedPkgs {}

