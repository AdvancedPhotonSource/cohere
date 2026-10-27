ARG CUDA_VERSION=13.0
ARG PYTORCH_VERSION=2.12.0
ARG CUDNN_VERSION=9

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    QT_X11_NO_MITSHM=1

WORKDIR /app
COPY . /src

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        openmpi-bin \
        libopenmpi-dev \
        libqt6gui6 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libx11-xcb1 \
        libgl1 \
        libegl1 \
        libopengl0 \
        libdbus-1-3 \
        libnss3 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxi6 \
        libxrender1 \
        libxtst6 \
        libsm6 \
        libice6 \
        libfontconfig1 \
        xauth \
        x11-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --root-user-action=ignore --no-cache-dir --upgrade pip \
    && MPICC=/usr/bin/mpicc python3 -m pip install \
        --break-system-packages \
        --root-user-action=ignore \
        --no-cache-dir \
        --no-binary=mpi4py \
        mpi4py \
    && python3 -m pip install \
        --break-system-packages \
        --root-user-action=ignore \
        --no-cache-dir \
        numpy pyzmq cupy-cuda13x \
    && python3 -m pip install \
        --break-system-packages \
        --root-user-action=ignore \
        --no-cache-dir \
        /src \
    && python3 -m pip install \
        --break-system-packages \
        --root-user-action=ignore \
        --no-cache-dir \
        /src/cohere-ui \
    && python3 -m pip install \
        --break-system-packages \
        --root-user-action=ignore \
        --no-cache-dir \
        /src/cohere_beamlines \
    && rm -rf /src

CMD ["cohere_gui"]