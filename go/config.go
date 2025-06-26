package main

import (
	"errors"
	"log"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

// AppConfig contains the overall configuration.
type AppConfig struct {
	WebSocket WebSocketConfig `yaml:"websocket"`
	WebRTC    WebRTCConfig    `yaml:"webrtc"`
}

// WebSocketConfig contains configuration specific for the WebSocket.
type WebSocketConfig struct {
	Address       string        `yaml:"address"`
	RetryInterval time.Duration `yaml:"retryInterval"`
}

// WebRTCConfig contains the configuration specific for WebRTC (STUN/TURN).
type WebRTCConfig struct {
	StunServers []string    `yaml:"stunServers"`
	TurnServers []ICEServer `yaml:"turnServers"`
}

// ICEServer defines the structure for a single ICE server (STUN or TURN).
type ICEServer struct {
	URLs       []string `yaml:"urls"`
	Username   string   `yaml:"username,omitempty"`
	Credential string   `yaml:"credential,omitempty"`
}

// loadConfig loads the configuration from the specified YAML file.
// It returns an error if any required field is missing.
func loadConfig(configPath string) (*AppConfig, error) {
	absPath, err := filepath.Abs(configPath)
	if err != nil {
		return nil, err
	}
	log.Printf("Loading configuration from: %s", absPath)
	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, err
	}

	var cfg AppConfig
	err = yaml.Unmarshal(data, &cfg)
	if err != nil {
		return nil, err
	}

	// Check for required configuration fields.
	if cfg.WebSocket.Address == "" {
		return nil, errors.New("Missing configuration: WebSocket.address")
	}
	if cfg.WebSocket.RetryInterval == 0 {
		return nil, errors.New("Missing configuration: WebSocket.retryInterval")
	}
	if len(cfg.WebRTC.StunServers) == 0 {
		return nil, errors.New("Missing configuration: WebRTC.stunServers")
	}
	if len(cfg.WebRTC.TurnServers) == 0 {
		return nil, errors.New("Missing configuration: WebRTC.turnServers")
	}

	return &cfg, nil
}
