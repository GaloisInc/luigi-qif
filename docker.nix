let
  pkgs = import ./pinned_pkgs.nix;
  inputs = import ./pkgs.nix {pkgs=pkgs;};
in inputs // {
  vim = pkgs.vim;
  nano = pkgs.nano;
  coreutils = pkgs.coreutils;
  bash = pkgs.bash;
  curl = pkgs.curl;
  wget = pkgs.wget;
}

