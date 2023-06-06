ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV USER_NAME=user

RUN if [ -e "/etc/apt/sources.list.d/cuda.list" ]; then \
        mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.save; \
    fi
RUN if [ -e "/etc/apt/sources.list.d/nvidia-ml.list" ]; then \
        mv /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/nvidia-ml.list.save; \
    fi
RUN apt update && apt install -y sudo curl software-properties-common \
    libgl1-mesa-dev \
    cmake \
    freeglut3-dev \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}
RUN chmod u+s /usr/sbin/useradd \
    && chmod u+s /usr/sbin/groupadd

RUN git clone https://github.com/kosuke55/hanging_points_generator /hanging_points_generator\
    && cd /hanging_points_generator \
    && sed -i 's/sklearn/scikit-learn/g' setup.py \
    && pip install scikit-build \
    && pip install -e . \
    && pip install open3d pyglet==1.4.10

USER $USER_NAME
ENV HOME /home/$USER_NAME

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
