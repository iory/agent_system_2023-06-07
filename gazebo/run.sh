#!/bin/bash

xhost +local:root
docker run --rm \
       --gpus all \
       --privileged \
       --volume="/dev:/dev" \
       --name "agent_system_melodic" \
       --net=host \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="$HOME/.ros/data:/root/.ros/data:rw" \
       -it agent_system_melodic /bin/bash
xhost +local:docker
