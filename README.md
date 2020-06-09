# `luigi-qif`

`luigi-qif` helps users locate and _plumb_ information _leaks_ in their secure computation code.
Please see `manual.pdf` for details.

# Quickstart (with Docker)

1. Run `docker build -t tamba-luigi-qif .`
2. Run `docker run -it --rm tamba-luigi-qif`
3. In the resulting shell, you can run commands like `luigi-qif --priors examples/sum.priors --mpc examples/sum.mpc static-leakage`
4. If you would like to run the provided test-suite, you can execute `./run-tests.sh` in the resulting shell.

# Quickstart (with Nix)

1. Run `nix-shell --pure`.
2. In the resulting shell, you can run commands like `./luigi-qif.py --priors examples/sum.priors --mpc examples/sum.mpc static-leakage`
3. If you would like to run the provided test-suite, you can execute `./run-tests.sh` in the resulting shell.

# Acknowledgments

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA).

The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
