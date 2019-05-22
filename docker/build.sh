docker volume create --name etaler-volume
docker -D build  -f ../Dockerfile  --tag etaler:latest .

