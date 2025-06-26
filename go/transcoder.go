package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

// Frame represents a media frame along with its metadata.
type Frame struct {
	ClientID uint32 // Identifier of the client sending the frame.
	FrameLen uint32 // Length of the frame data.
	FrameNr  uint32 // Frame sequence number.
	Data     []byte // Actual frame data.
}

// Transcoder defines the interface for a transcoder converting raw frames.
type Transcoder interface {
	UpdateBitrate(bitrate uint32)
	UpdateProjection()
	EncodeFrame(data []byte, framecounter uint32, bitrate uint32) *Frame
	IsReady() bool
	GetEstimatedBitrate() uint32
	GetFrameCounter() uint32
	IncrementFrameCounter()
	NextFrame() []byte
}

// readFiles reads all files from a directory and returns a slice of file contents and their sizes.
func readFiles(directory string) ([][]byte, []int64, error) {
	var fileContents [][]byte
	var fileSizes []int64

	// Retrieve directory file list.
	files, err := ioutil.ReadDir(directory)
	if err != nil {
		return nil, nil, err
	}

	// Iterate over each file, reading its content and size.
	for _, file := range files {
		filePath := filepath.Join(directory, file.Name())

		// Read file content.
		content, err := os.ReadFile(filePath)
		if err != nil {
			return nil, nil, err
		}
		fileContents = append(fileContents, content)

		// Append file size.
		fileSizes = append(fileSizes, file.Size())
	}

	return fileContents, fileSizes, nil
}

// TranscoderSharedMemory implements the Transcoder interface using shared memory.
type TranscoderSharedMemory struct {
	isReady      bool         // Indicates if the transcoder is ready.
	frameCounter uint32       // Counts the frames processed.
	shm          SharedMemory // Reference to the shared memory instance.
	squid        bool         // Flag to trigger special processing.
	squidCounter uint32       // Counter for squid flag management.
}

// NewTranscoderSharedMemory creates a new instance of TranscoderSharedMemory.
func NewTranscoderSharedMemory() (*TranscoderSharedMemory, error) {
	shm := GetInstanceSharedMemory()
	if shm == nil {
		return nil, fmt.Errorf("failed to initialize shared memory")
	}
	return &TranscoderSharedMemory{isReady: true, frameCounter: 0, shm: *shm, squid: false, squidCounter: 0}, nil
}

// NextFrame retrieves the next frame from shared memory based on the provided target bitrate.
// It also manages the squid flag logic.
func (t *TranscoderSharedMemory) NextFrame(targetBitrate int) []byte {
	frame, err := t.shm.getFrame(targetBitrate, t.squid)
	// Manage the squid flag: reset after one use.
	if t.squidCounter >= 1 {
		t.squid = false
		t.squidCounter = 0
	} else {
		t.squidCounter++
	}
	if err != nil {
		fmt.Printf("Error reading frame: %v\n", err)
		return nil
	}
	return frame
}

// EncodeFrame creates a Frame object from raw frame data and frame counter.
func (t *TranscoderSharedMemory) EncodeFrame(data []byte, framecounter uint32) *Frame {
	if data == nil {
		return nil
	}
	// Create a new Frame with the provided data and frame counter.
	rFrame := Frame{
		ClientID: 0,                 // Set to 0 if not applicable.
		FrameLen: uint32(len(data)), // Frame length is determined by data length.
		FrameNr:  framecounter,
		Data:     data,
	}
	return &rFrame
}

// GetFrameCounter returns the number of frames processed so far.
func (t *TranscoderSharedMemory) GetFrameCounter() uint32 {
	return t.frameCounter
}

// IncrementFrameCounter increments the frame counter by one.
func (t *TranscoderSharedMemory) IncrementFrameCounter() {
	t.frameCounter++
}
