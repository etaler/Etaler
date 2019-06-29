FROM  ubuntu:18.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata g++  wget git opencl-headers \
    ocl-icd-opencl-dev libcereal-dev sudo nano cmake apt-utils \
    libtbb-dev gdb python python-dev python-matplotlib python-numpy python2.7-dev policykit-1 x11-apps xorg-dev libglu1-mesa-dev

