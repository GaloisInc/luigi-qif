FROM nixos/nix

COPY . /luigi-qif
WORKDIR /luigi-qif
RUN nix-env -f ./docker.nix -i '.*'
RUN echo '#!/bin/sh' > /usr/local/bin/luigi-qif && \
    echo 'exec python3 /luigi-qif/luigi-qif.py $@' >> /usr/local/bin/luigi-qif && \
    chmod +rx /usr/local/bin/luigi-qif
