# robot

This repository contains the suite of programs necessary to start and use Diane on the robot side.

## Configuration

The main configuration for the Go application is located in `./go/config/config.yaml`.

This file allows you to specify parameters for:

- **WebSocket**:  
  - `address`: The address and port of the WebSocket server to connect to (e.g., `"your.server.address:3001"`).  
  - `retryInterval`: The reconnection interval to the WebSocket server in case of disconnection (e.g., `"1s"`).

- **WebRTC**:  
  - `stunServers`: A list of STUN servers used for NAT negotiation (e.g., `"stun:stun.l.google.com:19302"`).  
  - `turnServers`: A list of TURN servers, which can include:  
    - `urls`: The URLs of the TURN server (e.g., `"turn:your.turn.server:3478"`).  
    - `username`: The username for authentication to the TURN server.  
    - `credential`: The credential for authentication to the TURN server.

Configuration Example:  
```yaml
# ./go/config/config.yaml
websocket:
  address: "your.server.address:3001"
  retryInterval: 1s

webrtc:
  stunServers:
    - "stun:stun.l.google.com:19302"
  turnServers:
    - urls:
        - "turn:your.turn.server:3478"
      username: "username"
      credential: "credential"
```

## Project Startup

To fully start the system, follow these steps in order:

1. **Start the ROS shm2ros node**  
   This is necessary for communication between ROS components and the rest of the system.  
   Follow the instructions in the **shm2ros** repository’s README.  
   ```bash
   # shm2ros Repository
   shm2ros/deploy_shm2ros.sh
   shm2ros/docker_run_deploy.sh
   ```

2. **Start the ZED camera**  
   Use the `make run-nvidia-sdk` target to launch a Docker container with the ZED SDK and required settings.  
   ```bash
   make run-nvidia-sdk
   python3 main.py
   ```

3. **Build the Go application (if not already done)**  
   If this is your first time running the Go application or if you’ve modified the source, build the Docker image from the project root:  
   ```bash
   make build-go
   ```

4. **Run the Go application**  
   Once the Docker image is built, start the Go application from the project root:  
   ```bash
   make run-go
   ```  
   This will run the Go application inside a Docker container, using the configuration in `./go/config/config.yaml` and mounting the log directory.

4. **(OPTIONAL) Run with Custom Server**
    To start the WebRTC client specifying a different signaling server address:

    ```bash
    make run-go 192.168.1.100:9203
    ```

    This command will temporarily override the address configured in `config.yaml`.