FROM stereolabs/zed:4.2-devel-cuda12.1-ubuntu22.04

RUN apt update && apt upgrade -y
RUN apt install python3 python3-venv -y

WORKDIR /workspace
RUN cat /usr/local/zed/get_python_api.py
RUN python3 -m venv /.venv \
    && bash -c "source /.venv/bin/activate \
        && pip install -U \
	    cython numpy opencv-python pyopengl requests DracoPy websockets torch"
RUN bash -c "source /.venv/bin/activate && cd /usr/local/zed/ && python3 get_python_api.py"
RUN apt install usbutils -y
RUN echo 'source /.venv/bin/activate' >> /root/.bashrc
