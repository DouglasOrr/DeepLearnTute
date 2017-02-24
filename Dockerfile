FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update -q && apt-get install -qy \
        g++-4.9                              \
        git-core                             \
        nodejs-legacy                        \
        npm                                  \
        python3                              \
        python3-dev                          \
        python3-matplotlib                   \
        python3-numpy                        \
        python3-pip                          \
        python3-scipy                        \
        wget                                 \
    && apt-get clean

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs

# OpenBLAS & various dependencies
RUN cd /tmp                                             \
    && git clone https://github.com/xianyi/OpenBLAS.git \
    && cd OpenBLAS                                      \
    && make DYNAMIC_ARCH=1 NO_AFFINITY=1 NUM_THREADS=32 \
    && make PREFIX=/opt/openblas install                \
    && pip3 install --upgrade numpy                     \
    && mkdir /usr/bin/gcc-for-nvcc                      \
    && ln -s /usr/bin/gcc-4.9 /usr/bin/gcc-for-nvcc/gcc \
    && echo "compiler-bindir = /usr/bin/gcc-for-nvcc/" >> /usr/local/cuda/bin/nvcc.profile \
    && npm install -g configurable-http-proxy

# Library & data
COPY . /tmp/dlt
RUN cd /tmp/dlt                                         \
    && pip3 install -r requirements.txt                 \
    && python3 setup.py install                         \
    && ./scripts/prepare_uji /data/uji /test/uji        \
    && mkdir -p /examples                               \
    && cp -r examples/* /examples                       \
    && mkdir -p /etc/jupyterhub                         \
    && cp scripts/jupyterhub_config.py /etc/jupyterhub/ \
    && echo "admin:istrator::::/home/admin:" | newusers
