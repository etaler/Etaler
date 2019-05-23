docker run   -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix   -it  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --mount source=etaler-volume,target=/home etaler:latest
