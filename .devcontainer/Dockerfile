FROM  ubuntu:18.04


USER root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata g++  wget git opencl-headers \
    ocl-icd-opencl-dev libcereal-dev sudo nano cmake apt-utils \
    libtbb-dev gdb python python-dev python-matplotlib python-numpy python2.7-dev policykit-1 x11-apps xorg-dev libglu1-mesa-dev

WORKDIR /home/
# catch2 testing framwork
RUN mkdir /usr/include/catch2 &&  wget https://github.com/catchorg/Catch2/releases/download/v2.7.2/catch.hpp -P /usr/include/catch2


RUN echo "#!/bin/bash" > easy3d_script.sh
RUN echo "git clone  --recurse-submodules  https://github.com/alior101/Easy3D.git" >> easy3d_script.sh
RUN echo "cd Easy3D && mkdir build && cd build && cmake .. && make" >> easy3d_script.sh
RUN chmod +x easy3d_script.sh
RUN /home/easy3d_script.sh

RUN echo "#!/bin/bash" > etaler_script.sh
RUN echo "git clone  --recurse-submodules  https://github.com/alior101/Etaler.git" >> etaler_script.sh
RUN echo "cd Etaler && mkdir build && cd build && cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DETALER_BUILD_TESTS=OFF .. && make" >> etaler_script.sh
RUN chmod +x etaler_script.sh
RUN /home/etaler_script.sh

WORKDIR /home/Etaler