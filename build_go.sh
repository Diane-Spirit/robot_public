#!/bin/bash

# Imposta il Dockerfile predefinito
DOCKERFILE=${1:-Dockerfile.GO}
# Costruisce l'immagine Docker
sudo docker build -t robot_go -f "$DOCKERFILE" .

