package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	// Define and parse the command-line flag for the WebSocket (signaling) server address.
	wsAddrFlag := flag.String("ws", "", "WebSocket server address (e.g., localhost:3001)")
	flag.Parse()

	// Load configuration from file.
	configFile := "./config/config.yaml"
	cfg, err := loadConfig(configFile)
	if err != nil {
		log.Fatalf("Error loading configuration from file '%s': %v", configFile, err)
	}

	// Use the flag value if provided; otherwise, fall back to the address from the config.
	var wsAddress string
	if *wsAddrFlag != "" {
		wsAddress = *wsAddrFlag
		log.Printf("Using signaling server address from flag: %s", wsAddress)
	} else {
		wsAddress = cfg.WebSocket.Address
		log.Printf("Using signaling server address from config: %s", wsAddress)
	}

	wsURL := "ws://" + wsAddress
	log.Printf("Target WebSocket server URL: %s", wsURL)
	retryInterval := cfg.WebSocket.RetryInterval

	var wsClient *WebSocketClient

	// Attempt to connect to the WebSocket server with retries.
	// This loop will continue until a connection is successfully established.
	for {
		log.Printf("Attempting to connect to WebSocket server at %s...", wsURL)
		// NewWebSocketClient attempts to dial the WebSocket server and returns a client instance or an error.
		currentWsClient, err := NewWebSocketClient(wsURL)
		if err == nil {
			// Connection to WebSocket server was successful.
			log.Println("Successfully connected to WebSocket server.")
			// Initialize the WebSocket client, which typically starts its message listening loop.
			currentWsClient.Init()
			log.Println("WebSocket client message listener started.")
			wsClient = currentWsClient
			break
		} else {
			// Connection failed. Log the error and wait before retrying.
			log.Printf("Failed to connect to WebSocket server: %v. Retrying in %s...", err, retryInterval)
			time.Sleep(retryInterval)
		}
	}

	// Initialize the WebRTC PeerConnection using the established WebSocket client.
	log.Println("Initializing PeerConnection...")
	peerConn := NewPeerConnection(wsClient)
	log.Println("PeerConnection initialized.")
	// Start the PeerConnection's message listening and event handling.
	peerConn.Init(cfg.WebRTC, false)

	log.Println("Starting media track process in background...")
	// startMediaTrack is expected to handle the setup and streaming of media.
	// It runs in a separate goroutine to avoid blocking the main flow.
	go peerConn.startMediaTrack()

	// Set up a channel to listen for OS interrupt signals (SIGINT, SIGTERM)
	// for graceful shutdown of the application.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	<-sigs
	log.Println("Shutdown signal received: closing connections...")

	// Close the WebSocket connection.
	// The WebSocketClient's Close method is expected to gracefully terminate
	// its connection and any associated goroutines (e.g., the message reader).
	if err := wsClient.Close(); err != nil {
		log.Printf("Error closing WebSocket client: %v", err)
	}

	// Close the WebRTC peer connection.
	if err := peerConn.webrtcConnection.Close(); err != nil {
		log.Printf("Error closing WebRTC peer connection: %v", err)
	}
	log.Println("Application terminated gracefully.")
}
