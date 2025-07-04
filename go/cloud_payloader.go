package main

import (
	"bytes"
	"encoding/binary"
)

// AV1Payloader payloads AV1 packets
type PointCloudPayloader struct {
	frameCounter uint32
}

// Payload fragments a AV1 packet across one or more byte arrays
// See AV1Packet for description of AV1 Payload Header
func (p *PointCloudPayloader) Payload(mtu uint16, payload []byte) (payloads [][]byte) {
	payloadDataOffset := uint32(0)
	payloadLen := uint32(len(payload))
	payloadRemaining := payloadLen
	for payloadRemaining > 0 {
		currentFragmentSize := uint32(len(FramePacket{}.Data))
		if payloadRemaining < currentFragmentSize {
			currentFragmentSize = payloadRemaining
		}
		p := NewFramePacket(p.frameCounter, payloadLen, currentFragmentSize, payloadDataOffset, payload)
		buf := new(bytes.Buffer)

		if err := binary.Write(buf, binary.LittleEndian, p); err != nil {
			panic(err)
		}
		payloads = append(payloads, buf.Bytes())
		payloadDataOffset += currentFragmentSize
		payloadRemaining -= currentFragmentSize
	}
	p.frameCounter++
	return payloads
}

func NewPointCloudPayloader() *PointCloudPayloader {
	return &PointCloudPayloader{0}
}
