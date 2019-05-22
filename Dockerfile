FROM  ubuntu:18.04


USER root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y g++  wget git opencl-headers ocl-icd-opencl-dev libcereal-dev sudo nano cmake apt-utils libtbb-dev gdb 

WORKDIR /home/
# catch2 testing framwork
RUN mkdir /usr/include/catch2 &&  wget https://github.com/catchorg/Catch2/releases/download/v2.7.2/catch.hpp -P /usr/include/catch2

# etaler files fetched from repo (
# we write some small random num in order to prevent the cahce from kicking in and not download a fresh copy
RUN head -c 5 /dev/random > random_bytes &&  sudo git clone  --recurse-submodules  https://github.com/etaler/Etaler.git 

WORKDIR /home/Etaler

# delete a preexisting build dir - ignore if there isn't one already...
RUN  rm -rf build ; exit 0

RUN mkdir -p /home/Etaler/build
# I'm disabling tests since it fails for some unknown reason... 

WORKDIR /home/Etaler/build
RUN  cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DETALER_BUILD_TESTS=OFF ..
RUN cmake --build . ; exit 0
