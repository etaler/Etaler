FROM  ubuntu:18.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata g++  wget git opencl-headers \
    ocl-icd-opencl-dev libcereal-dev sudo nano cmake apt-utils \
    libtbb-dev gdb policykit-1 x11-apps xorg-dev libglu1-mesa-dev \
    libglew-dev libglfw3-dev libncurses5 beignet clinfo

RUN wget https://github.com/catchorg/Catch2/releases/download/v2.7.2/catch.hpp -P /usr/include/
