# Building on Linux

Building Etaler on Linux should be easy as it is mainly developed on Linux.

## Using Docker

Etaler's repo ships with a Docerfile in the `.devcontainer` directory. You can copy the file into the docker folder and utilize docker for an easy build.

```shell
cd Etaler/docker
cp ../.devcontainer/Dockerfile .
# Build the library
docker -D build   --tag etaler:latest .
# Run the container
docker run --rm -it -e DISPLAY=:0 --cap-add=SYS_PTRACE --mount source=etaler-volume,target=/home etaler:latest
```

## Building locally

If you are like me - want to use the library locally on the system and/or want to deploy it to an embedded system, Docker may not be an option for you. No worries, building locally is also very easy.

Here I show how to setup your system. You'll need to adapt the code if you are not using Arch Linux.

### Installing dependency

```shell
sudo pacman -S gcc cmake catch2 cereal intel-tbb opencl-headers
```

### Clone and build

```shell
git clone https://github.com/Etaler/Etaler --recursive
cd Etaler
mkdir build && cd build
cmake ..
make -j4
```

## Using the AUR

Etaler is avaliable on AUR as [etaler-git](https://aur.archlinux.org/packages/etaler-git/). Use a AUR helper like `yay` and it will sort out the dependencies for you.

```shell
yay etaler-git
```
