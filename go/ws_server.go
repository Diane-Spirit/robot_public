package main

import (
	"log"
	"sync"

	"github.com/gorilla/websocket"
)

type CallbackSend func(error)
type CallbackReceive func([]byte, error)

// WebSocketClient encapsulates a WebSocket connection.
type WebSocketClient struct {
	conn      *websocket.Conn
	mu        sync.Mutex
	callbacks []CallbackReceive
	cbMu      sync.RWMutex
	stopCh    chan struct{}
	wg        sync.WaitGroup
}

// NewWebSocketClient establishes a connection to the specified URL and returns a new WebSocketClient.
// An error is returned if the connection cannot be established.
func NewWebSocketClient(url string) (*WebSocketClient, error) {
	c, resp, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		if resp != nil {
			log.Printf("HTTP Status: %d", resp.StatusCode)
		}
		return nil, err
	}
	return &WebSocketClient{
		conn:      c,
		callbacks: make([]CallbackReceive, 0),
		stopCh:    make(chan struct{}),
	}, nil
}

// RegisterCallback registers a callback that is invoked whenever a message is received.
func (w *WebSocketClient) RegisterCallback(callback CallbackReceive) {
	w.cbMu.Lock()
	defer w.cbMu.Unlock()
	w.callbacks = make([]CallbackReceive, 0)
	w.callbacks = append(w.callbacks, callback)
}

// Init starts a goroutine that continuously reads messages from the WebSocket connection.
// All registered callbacks are invoked with the received message or an error.
// The goroutine terminates if an error occurs or when the client is closed.
func (w *WebSocketClient) Init() {
	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		for {
			select {
			case <-w.stopCh:
				return
			default:
				_, msg, err := w.conn.ReadMessage()
				w.cbMu.RLock()
				for _, cb := range w.callbacks {
					cb(msg, err)
				}
				w.cbMu.RUnlock()
				if err != nil {
					return
				}
			}
		}
	}()
}

// SendMessage transmits a message over the WebSocket connection.
// If the message cannot be sent, the error is logged.
func (w *WebSocketClient) SendMessage(message string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	err := w.conn.WriteMessage(websocket.TextMessage, []byte(message))
	if err != nil {
		log.Printf("Error sending message: %v", err)
	}
}

// Close terminates the WebSocket connection and waits for the reading goroutine to finish.
func (w *WebSocketClient) Close() error {
	close(w.stopCh)
	err := w.conn.Close()
	w.wg.Wait()
	return err
}
