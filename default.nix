let
  pkgs = import ./pinned_pkgs.nix;
  inputs = import ./pkgs.nix {pkgs=pkgs;};
in with pkgs;
mkShell rec {
  buildInputs = builtins.attrValues inputs;
}

