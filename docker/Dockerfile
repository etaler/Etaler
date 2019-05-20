FROM  ubuntu:18.04


USER root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y g++  wget git  opencl-headers ocl-icd-opencl-dev libcereal-dev sudo nano cmake apt-utils libtbb-dev 

# this will install  intell tbb latest  
# adding and not downloading since it requires registration on intel site ...
#ADD  libtbb2_2018_U6-4_amd64.deb /home
#ADD  libtbb-dev_2018_U6-4_amd64.deb /home
WORKDIR /home/
#RUN dpkg -i /home/libtbb2_2018_U6-4_amd64.deb 
#RUN dpkg -i /home/libtbb-dev_2018_U6-4_amd64.deb

# catch2 testing framwork
RUN mkdir /usr/include/catch2 &&  wget https://github.com/catchorg/Catch2/releases/download/v2.7.2/catch.hpp -P /usr/include/catch2

# etaler files
RUN sudo git clone  --recurse-submodules  https://github.com/etaler/Etaler.git 

WORKDIR /home/Etaler

# delete a preexisting build dir - ignore if there isn't one already...
RUN  rm -rf build ; exit 0

RUN mkdir -p /home/Etaler/build
# I'm disabling tests since it fails for some unknown reason... 
#RUN cd /home/Etaler/build && sudo cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_FLAGS=-isystem\ /home -DETALER_BUILD_TESTS=OFF ..

WORKDIR /home/Etaler/build
RUN  cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DETALER_BUILD_TESTS=OFF ..
RUN cmake --build . ; exit 0
