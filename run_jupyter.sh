docker run --rm -p 8888:8888 \
    --gpus all \
    -v ./workspace:/home/jovyan/work/workspace \
    -v ./notebooks:/home/jovyan/work/notebooks \
     quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8