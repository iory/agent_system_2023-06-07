#!/bin/bash

if [ ! -d "hanging_points_generator" ]; then
    git clone https://github.com/kosuke55/hanging_points_generator
fi

if [ ! -f "hanging_points_generator/hanging_points_generator/urdf_for_hanging.tar.gz" ]; then
    FILEID=1UOFshBGTCC-dHruVTCLZlqEadVZ4G8kP
    wget "https://drive.google.com/uc?export=download&id=$FILEID" -O hanging_points_generator/hanging_points_generator/urdf_for_hanging.tar.gz
    cd hanging_points_generator/hanging_points_generator
    tar xvzf urdf_for_hanging.tar.gz
    cd -
fi

docker run --rm \
       -u "$(id -u $USER):$(id -g $USER)" \
       --userns=host \
       --gpus all \
       --shm-size=1g \
       --name hanging_pose_generator \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="$(pwd)/hanging_points_generator:/workspace/hanging_points_generator:rw" \
       -ti hanging_pose_generator /bin/bash
