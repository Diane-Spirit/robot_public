package main

import (
	"encoding/json"

	"github.com/pion/webrtc/v3"
)

// Robot represents a robot entity with a unique identifier and a name.
type Robot struct {
	ID   int    // Unique robot identifier assigned by the server.
	name string // Robot's name (used for registration purposes).
}

// NewRobot creates and returns a new Robot instance with the specified name.
func NewRobot(name string) *Robot {
	return &Robot{
		name: name,
	}
}

// registerRobot generates a JSON payload for robot registration over WebSocket.
// The payload includes the robot's name and the SDP offer. This data is sent
// to the signaling server to initiate the WebRTC connection setup.
func (r *Robot) registerRobot(offer webrtc.SessionDescription) (string, error) {
	payload := struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		SdpOffer string `json:"sdpOffer"`
	}{
		Type:     "register",
		Name:     r.name,
		SdpOffer: offer.SDP,
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return string(b), nil
}


func (r *Robot) updateRobot(offer webrtc.SessionDescription) (string, error) {
    payload := struct {
        Type     string `json:"type"`
        RobotId  int    `json:"robotId"`
        SdpOffer string `json:"sdpOffer"`
    }{
        Type:     "offer",
        RobotId:  r.ID,
        SdpOffer: offer.SDP,
    }

    b, err := json.Marshal(payload)
    if err != nil {
        return "", err
    }
    return string(b), nil
}

// candidateMessage constructs a JSON payload containing the ICE candidate information.
// This payload is used to communicate network candidates between peers during the
// ICE negotiation process.
func (r *Robot) candidateMessage(c *webrtc.ICECandidate) (string, error) {
	candidate := c.ToJSON().Candidate

	payload := struct {
		Type      string `json:"type"`
		RobotId   int    `json:"robotId"`
		Candidate string `json:"candidate"`
	}{
		Type:      "candidate",
		RobotId:   r.ID,
		Candidate: candidate,
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
