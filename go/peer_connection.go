package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"sync"
	"time"

	_ "net/http/pprof"

	"log"
	"strconv"

	"github.com/beevik/ntp"
	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/cc"
	"github.com/pion/interceptor/pkg/gcc"
	"github.com/pion/interceptor/pkg/nack"
	"github.com/pion/interceptor/pkg/twcc"
	"github.com/pion/sdp/v3"
	"github.com/pion/webrtc/v3"
)

// PeerConnectionFrame encapsulates the metadata and data buffer of a media frame.
type PeerConnectionFrame struct {
	ClientID   uint64
	FrameNr    uint32
	FrameLen   uint32
	CurrentLen uint32
	FrameData  []byte
}

// NewPeerConnectionFrame creates and returns a new media frame structure.
func NewPeerConnectionFrame(clientID uint64, frameNr, frameLen uint32) *PeerConnectionFrame {
	return &PeerConnectionFrame{
		ClientID:   clientID,
		FrameNr:    frameNr,
		FrameLen:   frameLen,
		CurrentLen: 0,
		FrameData:  make([]byte, frameLen),
	}
}

// IsComplete checks if the entire frame has been received.
func (pf *PeerConnectionFrame) IsComplete() bool {
	return pf.CurrentLen == pf.FrameLen
}

// PeerConnection encapsulates WebRTC state and related objects.
type PeerConnection struct {
	websocketConnection    *WebSocketClient
	webrtcConnection       *webrtc.PeerConnection
	clientID               uint64
	candidatesMux          sync.Mutex
	pendingCandidates      []*webrtc.ICECandidate
	estimator              cc.BandwidthEstimator
	track                  *TrackLocalCloudRTP
	api                    *webrtc.API
	frames                 map[uint32]*PeerConnectionFrame
	completedFramesChannel *RingChannel
	isReady                bool
	bandwidthDC            *webrtc.DataChannel
	bw                     int
	sendFlag               bool
	stopMediaTrackCh       chan struct{}

	transcoder *TranscoderSharedMemory

	robot        *Robot
	sharedMemory *SharedMemory

	currentFrameNr        uint64
	controlMessageCounter int32
	rtcCfg                WebRTCConfig

	reconnectMutex sync.Mutex
	rtpSender      *webrtc.RTPSender
}

// ControlMessage defines a structure for control commands.
type ControlMessage struct {
	Linear  Vector3 `json:"linear"`
	Angular Vector3 `json:"angular"`
}

// NewPeerConnection initializes a new PeerConnection instance using an existing WebSocketClient.
func NewPeerConnection(websocketConnection *WebSocketClient) *PeerConnection {
	robot := NewRobot("robot")
	sharedMemory := GetInstanceSharedMemory()

	return &PeerConnection{
		websocketConnection:    websocketConnection,
		candidatesMux:          sync.Mutex{},
		pendingCandidates:      make([]*webrtc.ICECandidate, 0),
		frames:                 make(map[uint32]*PeerConnectionFrame),
		completedFramesChannel: NewRingChannel(0),
		currentFrameNr:         0,
		robot:                  robot,
		sharedMemory:           sharedMemory,
		controlMessageCounter:  0,
		bw:                     10 * 1000000,
		sendFlag:               false,
		stopMediaTrackCh:       make(chan struct{}),
	}
}

// Init configures the WebRTC PeerConnection, data channels, and media track.
func (pc *PeerConnection) Init(rtcCfg WebRTCConfig, updateConn bool) {
	pc.initApi()
	pc.rtcCfg = rtcCfg

	// Configure ICE servers.
	var iceServers []webrtc.ICEServer
	for _, stunURL := range rtcCfg.StunServers {
		iceServers = append(iceServers, webrtc.ICEServer{URLs: []string{stunURL}})
	}
	for _, turnCfg := range rtcCfg.TurnServers {
		iceServers = append(iceServers, webrtc.ICEServer{
			URLs:       turnCfg.URLs,
			Username:   turnCfg.Username,
			Credential: turnCfg.Credential,
		})
	}

	config := webrtc.Configuration{ICEServers: iceServers}
	webrtcConn, err := pc.api.NewPeerConnection(config)
	if err != nil {
		panic(err)
	}
	pc.webrtcConnection = webrtcConn

	// Register connection callbacks.
	webrtcConn.OnDataChannel(pc.OnDataChannelCb)
	webrtcConn.OnICECandidate(pc.OnIceCandidateCb)
	webrtcConn.OnConnectionStateChange(pc.OnConnectionStateChangeCb)
	pc.websocketConnection.RegisterCallback(pc.websocketMessageHandler)

	// Create and initialize control Data Channel.
	_, err = pc.createControlDC()
	if err != nil {
		panic(err)
	}
	// Configure media track.
	codecCap := getCodecCapability()
	codecCap.RTCPFeedback = nil
	videoTrack, err := NewTrackLocalCloudRTP(codecCap, "video", "pion")
	if err != nil {
		panic(err)
	}
	pc.track = videoTrack

	// Add the media track to the PeerConnection.
	rtpSender, err := webrtcConn.AddTrack(videoTrack)
	if err != nil {
		panic(err)
	}

	pc.rtpSender = rtpSender

	// Start RTCP handling to receive feedback.
	go func(sender *webrtc.RTPSender, conn *webrtc.PeerConnection, stopChan chan struct{}) {
		rtcpBuf := make([]byte, 1500)
		for {
			select {
			case <-stopChan: // Listen for stop signal specific to this RTCP loop if needed, or rely on sender.Read error
				log.Println("RTCP read loop: Stop signal received (indirectly via connection close).")
				return
			default:
				if sender == nil || conn == nil || conn.ConnectionState() == webrtc.PeerConnectionStateClosed {
					log.Println("RTCP read loop: RTP sender or connection is nil/closed, exiting.")
					return
				}
				// SetReadDeadline can be useful here to prevent indefinite blocking
				// if err := sender.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
				// 	log.Printf("RTCP read loop: Error setting read deadline: %v", err)
				// 	return
				// }
				n, _, readErr := sender.Read(rtcpBuf)
				if readErr != nil {
					if readErr == io.EOF || readErr == io.ErrClosedPipe || readErr.Error() == "read: connection reset by peer" { // Common errors on close
						log.Printf("RTCP read loop: Connection closed or EOF: %v. Exiting.", readErr)
					} else {
						log.Printf("RTCP read error: %v. Exiting RTCP read loop.", readErr)
					}
					return
				}
				if n > 0 {
					// Process RTCP packet if necessary, though often just reading is enough for Pion's interceptors
					// log.Printf("RTCP read loop: Received %d bytes", n)
				}
			}
		}
	}(pc.rtpSender, pc.webrtcConnection, pc.stopMediaTrackCh)

	// Set local SDP offer and signal the remote side.
	offer, err := webrtcConn.CreateOffer(nil)
	if err != nil {
		panic(err)
	}
	if err := webrtcConn.SetLocalDescription(offer); err != nil {
		panic(err)
	}
	var message string

	if !updateConn {
		message, err = pc.robot.registerRobot(offer)
		if err != nil {
			panic(err)
		}
	} else {
		message, err = pc.robot.updateRobot(offer)
		if err != nil {
			panic(err)
		}
	}

	log.Printf("Sending %s message to server", map[bool]string{true: "update offer", false: "register offer"}[!updateConn])
	pc.websocketConnection.SendMessage(message)
}

// Close cleans up resources associated with the PeerConnection.
func (pc *PeerConnection) Close() {
	log.Println("Closing PeerConnection resources...")

	// Signal the media track to stop BEFORE closing the WebRTC connection
	// as startMediaTrack might use pc.estimator which comes from the connection.
	pc.StopMediaTrack() // This closes pc.stopMediaTrackCh

	if pc.webrtcConnection != nil {
		// Unregister callbacks to prevent them from firing during/after close
		// Note: Pion's PeerConnection doesn't have explicit Unregister methods.
		// Closing the connection should stop event propagation.

		if err := pc.webrtcConnection.Close(); err != nil {
			log.Printf("Error closing WebRTC connection: %v", err)
		}
		pc.webrtcConnection = nil // Nullify to prevent reuse
	}

	pc.candidatesMux.Lock()
	pc.pendingCandidates = make([]*webrtc.ICECandidate, 0) // Clear pending candidates
	pc.candidatesMux.Unlock()

	pc.isReady = false
	pc.bandwidthDC = nil
	pc.estimator = nil // Estimator is tied to the PeerConnection via interceptor
	pc.track = nil
	pc.rtpSender = nil

	// Do not close pc.websocketConnection here as it's managed externally
	log.Println("PeerConnection resources closed.")
}

// attemptReconnect tries to re-establish a failed PeerConnection.
func (pc *PeerConnection) attemptReconnect() {
	pc.reconnectMutex.Lock() // Ensure only one reconnect attempt at a time
	// Check if already reconnected by another thread or if connection is not in a failed state
	if pc.isReady && pc.webrtcConnection != nil && pc.webrtcConnection.ConnectionState() == webrtc.PeerConnectionStateConnected {
		log.Println("attemptReconnect: Connection is already established. Aborting this attempt.")
		pc.reconnectMutex.Unlock()
		return
	}
	log.Println("attemptReconnect: Starting reconnection process...")
	pc.reconnectMutex.Unlock() // Unlock early if the rest of the function is long, or defer

	// 1. Cleanly close the current (failed) connection and its resources
	pc.Close()

	// Brief pause to allow goroutines to shut down. Consider more robust synchronization if needed.
	time.Sleep(500 * time.Millisecond)

	pc.reconnectMutex.Lock() // Re-lock for modifying pc state
	defer pc.reconnectMutex.Unlock()

	// 2. Reset necessary fields on the *existing* pc instance
	pc.isReady = false
	pc.frames = make(map[uint32]*PeerConnectionFrame)
	pc.completedFramesChannel = NewRingChannel(0)
	pc.currentFrameNr = 0
	// pc.controlMessageCounter // Decide if this should be reset
	pc.stopMediaTrackCh = make(chan struct{}) // Create a new channel for the new media track

	// 3. Re-initialize the PeerConnection.
	//    pc.robot.ID should persist from the previous session if registration was successful.
	isUpdatingExistingRobot := (pc.robot.ID != 0)
	log.Printf("attemptReconnect: Re-initializing PeerConnection. Robot ID: %d. Will send update: %t", pc.robot.ID, isUpdatingExistingRobot)

	// Init will call initApi, create new PC, data channels, track, set local offer, and send it.
	pc.Init(pc.rtcCfg, isUpdatingExistingRobot) // pc.rtcCfg should hold the original config

	// 4. Restart the media track.
	//    It's important that startMediaTrack is called for the re-initialized pc.
	log.Println("attemptReconnect: Restarting media track.")
	go pc.startMediaTrack()

	log.Println("attemptReconnect: Reconnection process initiated.")
}

// OnDataChannelCb processes new Data Channels by setting proper message handlers.
func (pc *PeerConnection) OnDataChannelCb(dc *webrtc.DataChannel) {
	log.Printf("New DataChannel %q %d", dc.Label(), dc.ID())

	if dc.Label() == "bandwidth" {
		pc.bandwidthDC = dc

		// Setup callbacks on the bandwidth channel.
		dc.OnOpen(func() {
			log.Printf("Data channel %q %d opened", dc.Label(), dc.ID())
			log.Println("Bandwidth channel is open and ready for communication.")
		})
		dc.OnMessage(func(msg webrtc.DataChannelMessage) {
			if len(msg.Data) != 4 {
				log.Println("Invalid message length for bandwidth channel")
				return
			}
			pc.bw = int(binary.LittleEndian.Uint32(msg.Data[:]))
		})
		dc.OnClose(func() {
			log.Printf("Data channel %q %d closed", dc.Label(), dc.ID())
			pc.bandwidthDC = nil
		})
	} else if dc.Label() == "controltrack" {
		dc.OnMessage(pc.onMessageControlTrackCb)
	}
}

func (pc *PeerConnection) createControlDC() (*webrtc.DataChannel, error) {
	controlDC, err := pc.webrtcConnection.CreateDataChannel("controltrack", nil)
	if err != nil {
		return nil, err
	}
	controlDC.OnOpen(func() {
		log.Println("Control data channel opened")
	})
	controlDC.OnMessage(
		func(msg webrtc.DataChannelMessage) {
			var controlMsg ControlMessage
			if err := json.Unmarshal(msg.Data, &controlMsg); err != nil {
				log.Printf("Error unmarshalling control message: %v", err)
				return
			}

			// If the message is flagged as a squid test command.
			if controlMsg.Linear.Z != 0 {
				pc.transcoder.squid = true
				pc.transcoder.squidCounter = 0
				return
			}
			// Otherwise, write control data to shared memory.
			if err := pc.sharedMemory.writeControl(pc.controlMessageCounter, controlMsg.Linear, controlMsg.Angular); err != nil {
				log.Printf("Error writing control data to shared memory: %v", err)
				return
			}
			pc.controlMessageCounter++
		})
	return controlDC, nil
}

// onMessageControlTrackCb handles messages from the control track channel.
func (pc *PeerConnection) onMessageControlTrackCb(msg webrtc.DataChannelMessage) {
	messageData := string(msg.Data)
	switch messageData {
	case "stop":
		pc.sendFlag = false
	case "start":
		pc.sendFlag = true
	default:
		log.Println("Received unknown controltrack message")
	}
}

// SetRemoteDescription applies the remote SDP and sends any pending ICE candidates.
func (pc *PeerConnection) SetRemoteDescription(answer webrtc.SessionDescription) error {
	if err := pc.webrtcConnection.SetRemoteDescription(answer); err != nil {
		return err
	}
	pc.candidatesMux.Lock()
	defer pc.candidatesMux.Unlock()
	for _, c := range pc.pendingCandidates {
		message, err := pc.robot.candidateMessage(c)
		if err != nil {
			log.Printf("Error creating candidate message: %v", err)
		}
		pc.websocketConnection.SendMessage(message)
	}
	return nil
}

// AddICECandidate adds a new ICE candidate.
func (pc *PeerConnection) AddICECandidate(candidate string) error {
	err := pc.webrtcConnection.AddICECandidate(webrtc.ICECandidateInit{Candidate: candidate})
	if err != nil {
		log.Printf("Error adding ICE candidate: %v", err)
	}
	return err
}

// SetEstimator sets the congestion control bandwidth estimator.
func (pc *PeerConnection) SetEstimator(estimator cc.BandwidthEstimator) {
	pc.estimator = estimator
}

// OnIceCandidateCb handles ICE candidate generation and transmission.
func (pc *PeerConnection) OnIceCandidateCb(c *webrtc.ICECandidate) {
	if c == nil {
		return
	}
	pc.candidatesMux.Lock()
	defer pc.candidatesMux.Unlock()
	if pc.webrtcConnection.RemoteDescription() == nil {
		pc.pendingCandidates = append(pc.pendingCandidates, c)
	} else {
		message, err := pc.robot.candidateMessage(c)
		if err != nil {
			log.Printf("Error creating candidate message: %v", err)
			return
		}
		pc.websocketConnection.SendMessage(message)
	}
}

func (pc *PeerConnection) OnConnectionStateChangeCb(state webrtc.PeerConnectionState) {
	log.Printf("Peer connection state changed: %s", state.String())
	switch state {
	case webrtc.PeerConnectionStateDisconnected:
		pc.isReady = false
		// Disconnected can sometimes recover or might lead to Failed.
		// Consider if reconnect logic should trigger here or wait for Failed.
		log.Println("PeerConnectionStateDisconnected: Attempting reconnect.")
		go pc.attemptReconnect() // Or signal a manager
	case webrtc.PeerConnectionStateFailed:
		pc.isReady = false
		log.Println("PeerConnectionStateFailed: Attempting reconnect.")
		go pc.attemptReconnect() // Or signal a manager
	case webrtc.PeerConnectionStateConnected:
		pc.isReady = true
		log.Println("PeerConnectionStateConnected: Connection established.")
		// If there were pending candidates because remote description was late, send them now.
		pc.candidatesMux.Lock()
		if len(pc.pendingCandidates) > 0 {
			log.Printf("Connection now connected, sending %d pending ICE candidates.", len(pc.pendingCandidates))
			for _, cand := range pc.pendingCandidates {
				message, err := pc.robot.candidateMessage(cand)
				if err != nil {
					log.Printf("Error creating candidate message for pending candidate: %v", err)
					continue
				}
				pc.websocketConnection.SendMessage(message)
			}
			pc.pendingCandidates = make([]*webrtc.ICECandidate, 0) // Clear after sending
		}
		pc.candidatesMux.Unlock()

	case webrtc.PeerConnectionStateClosed:
		pc.isReady = false
		log.Println("PeerConnectionStateClosed: Connection has been closed.")
		// This state is usually final. No automatic reconnect from here unless explicitly desired.
	case webrtc.PeerConnectionStateNew:
		log.Println("PeerConnectionStateNew: Connection is new.")
	case webrtc.PeerConnectionStateConnecting:
		log.Println("PeerConnectionStateConnecting: Connection is attempting to connect.")
	}
}

// GetBitrate returns the current target bitrate.
func (pc *PeerConnection) GetBitrate() uint32 {
	return uint32(pc.estimator.GetTargetBitrate())
}

// GetFrameCounter returns the current frame counter.
func (pc *PeerConnection) GetFrameCounter() uint32 {
	return uint32(pc.currentFrameNr)
}

// SendFrame sends the provided frame via the local media track.
func (pc *PeerConnection) SendFrame(frame *Frame) {
	if frame != nil {
		go func(f *Frame) {
			pc.track.WriteFrame(f)
		}(frame)
		select {
		case pc.completedFramesChannel.In() <- frame:
		default:
		}
	}
	pc.currentFrameNr++
}

// initApi configures the WebRTC API and associated interceptors.
func (pc *PeerConnection) initApi() {
	settingEngine := webrtc.SettingEngine{}
	settingEngine.SetSCTPMaxReceiveBufferSize(16 * 1024 * 1024)

	i := &interceptor.Registry{}
	m := NewMediaEngine()

	// Create congestion controller for send-side bandwidth estimation.
	congestionController, err := cc.NewInterceptor(func() (cc.BandwidthEstimator, error) {
		return gcc.NewSendSideBWE(
			gcc.SendSideBWEMinBitrate(50_000*8),
			gcc.SendSideBWEInitialBitrate(10_000_000),
			gcc.SendSideBWEMaxBitrate(300_000_000),
		)
	})
	if err != nil {
		panic(err)
	}
	congestionController.OnNewPeerConnection(func(id string, estimator cc.BandwidthEstimator) {
		pc.estimator = estimator
	})
	i.Add(congestionController)

	if err = webrtc.ConfigureTWCCHeaderExtensionSender(m, i); err != nil {
		panic(err)
	}

	responder, _ := nack.NewResponderInterceptor()
	i.Add(responder)

	generator, err := twcc.NewSenderInterceptor(twcc.SendInterval(10 * time.Millisecond))
	if err != nil {
		panic(err)
	}
	i.Add(generator)

	nackGenerator, _ := nack.NewGeneratorInterceptor()
	i.Add(nackGenerator)

	pc.api = webrtc.NewAPI(
		webrtc.WithSettingEngine(settingEngine),
		webrtc.WithInterceptorRegistry(i),
		webrtc.WithMediaEngine(m),
	)
}

// NewMediaEngine creates and configures a new MediaEngine instance.
func NewMediaEngine() *webrtc.MediaEngine {
	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		panic(err)
	}
	if err := m.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: getCodecCapability(),
		PayloadType:        5,
	}, webrtc.RTPCodecTypeVideo); err != nil {
		panic(err)
	}
	// Video feedback registration.
	m.RegisterFeedback(webrtc.RTCPFeedback{Type: "nack"}, webrtc.RTPCodecTypeVideo)
	m.RegisterFeedback(webrtc.RTCPFeedback{Type: "nack", Parameter: "pli"}, webrtc.RTPCodecTypeVideo)
	m.RegisterFeedback(webrtc.RTCPFeedback{Type: webrtc.TypeRTCPFBTransportCC}, webrtc.RTPCodecTypeVideo)
	if err := m.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: sdp.TransportCCURI}, webrtc.RTPCodecTypeVideo); err != nil {
		panic(err)
	}

	// Audio feedback registration.
	m.RegisterFeedback(webrtc.RTCPFeedback{Type: webrtc.TypeRTCPFBTransportCC}, webrtc.RTPCodecTypeAudio)
	if err := m.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: sdp.TransportCCURI}, webrtc.RTPCodecTypeAudio); err != nil {
		panic(err)
	}
	return m
}

// getCodecCapability returns a predefined RTP codec configuration.
func getCodecCapability() webrtc.RTPCodecCapability {
	videoRTCPFeedback := []webrtc.RTCPFeedback{
		{Type: "goog-remb", Parameter: ""},
		{Type: "ccm", Parameter: "fir"},
		{Type: "nack", Parameter: ""},
		{Type: "nack", Parameter: "pli"},
	}
	return webrtc.RTPCodecCapability{
		MimeType:     "video/pcm",
		ClockRate:    90000,
		Channels:     0,
		SDPFmtpLine:  "",
		RTCPFeedback: videoRTCPFeedback,
	}
}

// startMediaTrack continuously captures, processes, and transmits media frames.
func (pc *PeerConnection) startMediaTrack() {
	timestamp := time.Now().Format("20060102_150405")
	filePath := fmt.Sprintf("./log/log_robot%d_%s.csv", pc.robot.ID, timestamp)
	header := []string{"Timestamp", "vLossRate", "vDelayRate", "vLoss", "state", "usage", "GCCBandwidth(b/s)", "ReceiveEstimateBandwidth(b/s)", "FrameDimension(B)"}
	csvFile, err := NewManagedCSV(filePath, header, true)
	if err != nil {
		log.Printf("Error initializing CSV logger: %v", err)
		return
	}

	log.Println("Starting media track")
	pc.transcoder, err = NewTranscoderSharedMemory()
	if err != nil {
		panic(err)
	}

	// Synchronize local time with NTP server time.
	timeFromNTP, err := ntp.Time("pool.ntp.org")
	if err != nil {
		panic(err)
	}
	offset := timeFromNTP.Sub(time.Now())

	// Main loop: fetch, process, and send frames.
	for {
		select {
		case <-pc.stopMediaTrackCh: 
			log.Println("Stopping media track loop.")
			return
		default:
			if pc != nil && pc.estimator != nil {
				frame := pc.transcoder.NextFrame(pc.estimator.GetTargetBitrate())
				if pc.sendFlag && pc.isReady {
					dimFrame := len(frame)
					encodedFrame := pc.transcoder.EncodeFrame(frame, pc.GetFrameCounter())
					pc.SendFrame(encodedFrame)
					if pc.transcoder.GetFrameCounter()%100 == 0 {
						runtime.GC()
					}

					vLossRate, _ := pc.estimator.GetStats()["lossTargetBitrate"].(int)
					vDelayRate, _ := pc.estimator.GetStats()["delayTargetBitrate"].(int)
					vLoss, _ := pc.estimator.GetStats()["averageLoss"].(float64)
					ts := float64(time.Now().Add(offset).UnixNano()) / 1e9
					state := pc.estimator.GetStats()["state"].(string)
					usage := pc.estimator.GetStats()["usage"].(string)
					targetBitrate := pc.estimator.GetTargetBitrate() / 1000000

					log.Println("------")
					log.Printf("Timestamp       : %.10f", ts)
					log.Printf("GCC Bandwidth   : %d Mbit/s", targetBitrate)
					log.Printf("Client Bandwidth: %d Mbit/s", pc.bw/1000000)
					log.Printf("vLossRate       : %d", vLossRate)
					log.Printf("vDelayRate      : %d", vDelayRate)
					log.Printf("vLoss           : %.2f", vLoss)
					log.Printf("State           : %s", state)
					log.Printf("Usage           : %s", usage)
					log.Printf("Frame Dimension : %d bytes", dimFrame)
					log.Println("------")

					csvFile.AppendData([]string{
						strconv.FormatFloat(ts, 'f', 10, 64),
						strconv.Itoa(vLossRate),
						strconv.Itoa(vDelayRate),
						strconv.FormatFloat(vLoss, 'f', 4, 64),
						state,
						usage,
						strconv.Itoa(int(pc.estimator.GetTargetBitrate())),
						strconv.Itoa(int(pc.bw)),
						strconv.Itoa(dimFrame),
					})
				}
				pc.transcoder.IncrementFrameCounter()
			}
		}
	}
}

func (pc *PeerConnection) StopMediaTrack() {
	pc.reconnectMutex.Lock() // Protect access to stopMediaTrackCh if it can be closed/recreated
	defer pc.reconnectMutex.Unlock()

	// Check if channel is already closed to prevent panic
	select {
	case <-pc.stopMediaTrackCh:
		log.Println("StopMediaTrack: Channel already closed.")
	default:
		log.Println("StopMediaTrack: Closing channel to stop media track.")
		close(pc.stopMediaTrackCh)
	}
}

// websocketMessageHandler handles messages received via the WebSocket.
func (pc *PeerConnection) websocketMessageHandler(msg []byte, err error) {
	log.Println("Received message from server:")
	log.Println(string(msg))
	if err != nil {
		log.Printf("Error receiving message: %v", err)
		return
	}
	var base struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(msg, &base); err != nil {
		log.Printf("Error parsing WebSocket message: %v", err)
		return
	}
	switch base.Type {
	case "registered":
		var payload struct {
			Type    string `json:"type"`
			RobotId int    `json:"robotId"`
		}
		if err := json.Unmarshal(msg, &payload); err != nil {
			log.Printf("Error parsing 'registered' message: %v", err)
			return
		}
		log.Printf("Registration completed, Robot ID: %d", payload.RobotId)
		pc.robot.ID = payload.RobotId
	case "sdpAnswer":
		log.Println("Received SDP answer")
		var payload struct {
			Type      string `json:"type"`
			SdpAnswer string `json:"sdpAnswer"`
		}
		if err := json.Unmarshal(msg, &payload); err != nil {
			log.Printf("Error parsing SDP answer: %v", err)
			return
		}
		log.Printf("Applying SDP answer: %s", payload.SdpAnswer)
		pc.SetRemoteDescription(webrtc.SessionDescription{
			Type: webrtc.SDPTypeAnswer,
			SDP:  payload.SdpAnswer,
		})
	case "candidate":
		var payload struct {
			Type      string `json:"type"`
			Candidate string `json:"candidate"`
		}
		if err := json.Unmarshal(msg, &payload); err != nil {
			log.Printf("Error parsing candidate message: %v", err)
			return
		}
		log.Printf("Received ICE candidate: %s", payload.Candidate)
		pc.AddICECandidate(payload.Candidate)
	default:
		log.Printf("Unhandled message type: %s", base.Type)
	}
}
