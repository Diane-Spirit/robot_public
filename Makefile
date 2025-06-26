BUILD_TYPE = Release
IMAGE_TYPE = base
TEST_TIME_SECS = 10

CONTAINER_IMAGE := websocket-zedsdk
CONTAINER_IMAGE_NVIDIA := websocket-nvidia
CONTAINER_IMAGE_NVIDIA_SDK := websocket-nvidia-sdk
CONTAINER_DEV_IMAGE := $(CONTAINER_IMAGE)-dev
GO_CONTAINER_IMAGE := robot_go  
GO_DOCKERFILE := Dockerfile.GO
PERCENT := %
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

default: build

build-nvidia: ## Build release container
	@docker build \
		--tag $(CONTAINER_IMAGE_NVIDIA) \
		--file Dockerfile.jetson \
		.

build-nvidia-sdk: ## Build release container
	@docker build \
		--tag $(CONTAINER_IMAGE_NVIDIA_SDK) \
		--file Dockerfile.jetsonsdk \
		.

build: ## Build release container
	@docker build \
		--tag $(CONTAINER_IMAGE) \
		--file Dockerfile \
		.

run: ## Run a disposable development container
	@docker run \
		--interactive \
		--tty \
		--rm \
		--runtime nvidia \
		--gpus all \
		--privileged \
		--net host \
		--ipc host \
		-e DISPLAY=$(DISPLAY) \
		--volume /tmp/.X11-unix:/tmp/.X11-unix \
		--env QT_X11_NO_MITSHM=1 \
		--volume $(ROOT_DIR)/workspace:/workspace \
		$(CONTAINER_IMAGE) \
		bash

run-nvidia: ## Run a disposable development container
	@docker run \
		--interactive \
		--tty \
		--rm \
		--runtime nvidia \
		--gpus all \
		--privileged \
		--net host \
		--ipc host \
		-e DISPLAY=$(DISPLAY) \
		--volume /tmp/.X11-unix:/tmp/.X11-unix \
		--env QT_X11_NO_MITSHM=1 \
		--volume $(ROOT_DIR)/workspace:/workspace \
		$(CONTAINER_IMAGE_NVIDIA) \
		bash

run-nvidia-sdk: ## Run a disposable development container
	@docker run \
		--interactive \
		--tty \
		--rm \
		--runtime nvidia \
		--gpus all \
		--privileged \
		--net host \
		--ipc host \
		-e DISPLAY=$(DISPLAY) \
		--volume /tmp/.X11-unix:/tmp/.X11-unix \
		--env QT_X11_NO_MITSHM=1 \
		--volume $(ROOT_DIR)/workspace:/workspace \
		--volume /run/jtop.sock:/run/jtop.sock \
		--volume ./calibration/:/usr/local/zed/settings/ \
		--volume ./resources:/usr/local/zed/resources \
		$(CONTAINER_IMAGE_NVIDIA_SDK) \
		bash


build-go:
	@docker build -t $(GO_CONTAINER_IMAGE) -f $(GO_DOCKERFILE) .

run-go: ## Build e run del container Go
	@$(eval EXTRA_ARGS := $(filter-out run-go,$(MAKECMDGOALS)))
	@if [ -z "$(EXTRA_ARGS)" ]; then \
	  EXTRA_ARGS=""; \
	else \
	  EXTRA_ARGS="-ws $(EXTRA_ARGS)"; \
	fi; \
	docker run -it --rm \
		--name robot_go_webrtc \
		--net host \
		--ipc host \
		--privileged \
		--volume ./go/log/:/app/log/ \
		--volume ./go/config/:/app/config/ \
		$(GO_CONTAINER_IMAGE) $$EXTRA_ARGS


clean: ## Clean image artifacts
	-docker rmi $(CONTAINER_IMAGE)
	-docker rmi $(CONTAINER_IMAGE_NVIDIA)
	-docker rmi $(CONTAINER_IMAGE_NVIDIA_SDK)
	-docker rmi $(GO_CONTAINER_IMAGE)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

# Regola "catch-all" per evitare errori sugli obiettivi extra
%:
	@:
.PHONY: default run-go run run-nvidia run-nvidia-sdk build-nvidia build-nvidia-sdk clean help
