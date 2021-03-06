#!/usr/bin/env bash
set -e
# prototype is taken from  https://github.com/arviz-devs/arviz/blob/master/scripts/container.sh
SRC_DIR=${SRC_DIR:-`pwd`}
COVERAGE=${COVERAGE:-"--cov geoopt"}
PYTORCH=${PYTORCH:-"pytorch"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.6"}
# Build container for use of testing or notebook
if [[ $* == *--build* ]]; then
    echo "Building Docker Image"
    docker build \
        -t geoopt \
        -f ${SRC_DIR}/scripts/Dockerfile \
        --build-arg SRC_DIR=. ${SRC_DIR} \
        --build-arg PYTORCH=${PYTORCH} \
        --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
        --rm
fi

if [[ $* == *--test* ]]; then
    echo "Testing Geoopt"
    docker run --rm -it --user $(id -u):$(id -g) --mount type=bind,source="$(pwd)",target=/opt/geoopt/ geoopt:latest \
        bash -c "pytest --durations=0 -v tests/ ${COVERAGE}"
        if [[ ${COVERAGE} ]]; then sed -i 's@/opt/geoopt@'${SRC_DIR}'@g' "$(pwd)/.coverage"; fi
fi

if [[ $* == *--bash* ]]; then
    echo "Running Bash"
    docker run --rm -it --user $(id -u):$(id -g) --mount type=bind,source="$(pwd)",target=/opt/geoopt/ geoopt:latest bash
fi
