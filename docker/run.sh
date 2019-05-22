#  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined is for enabling debug inside the container
#docker run --rm  -it  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --mount source=etaler-volume,target=/home etaler:latest
docker run   -it  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --mount source=etaler-volume,target=/home etaler:latest
