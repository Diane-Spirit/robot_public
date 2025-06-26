package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"sync"
	"syscall"
	"time"
)

const (
	headerSize  = 3
	dataMaxSize = 1843200                  // Maximum data size (480x480x8 bytes)
	dataSize    = headerSize + dataMaxSize // Total mapped size for data buffer
	flagSize    = 6                        // Size of flag memory region
	controlSize = 52                       // Size of control memory region
)

// SharedMemory is a singleton used to manage shared memory and associated flag areas.
type SharedMemory struct {
	dataMem            []byte // Mapped memory for frame data
	flagReadMem        []byte // Mapped memory for flag read/write operations
	controlMem         []byte // Mapped memory for control data
	semaphore_ros_read []byte // Mapped memory for ROS semaphore operations
}

var instance *SharedMemory
var once sync.Once

// GetInstanceSharedMemory returns the single instance of SharedMemory, initializing it if necessary.
func GetInstanceSharedMemory() *SharedMemory {
	once.Do(func() {
		instance = &SharedMemory{}
		if err := instance.init(); err != nil {
			// Print error and invalidate instance in case of initialization failure.
			log.Printf("Error initializing shared memory: %v\n", err)
			instance = nil
		}
	})
	return instance
}

// init initializes the shared memory segments and mapping required for operation.
func (s *SharedMemory) init() error {
	// Open shared memory file containing frame data (read-only).
	dataFile, err := os.OpenFile("/dev/shm/shared_pc", os.O_RDONLY, 0600)
	if err != nil {
		return fmt.Errorf("error opening shared_pc: %w", err)
	}
	defer dataFile.Close()

	// Open semaphore file used for read flags with read/write access.
	flagRead, err := os.OpenFile("/dev/shm/semaphore_read", os.O_RDWR, 0600)
	if err != nil {
		return fmt.Errorf("error opening semaphore_read: %w", err)
	}
	defer flagRead.Close()

	// Open semaphore file used for ROS reading with read/write access.
	semaphoreROS, err := os.OpenFile("/dev/shm/semaphore_ros_read", os.O_RDWR, 0600)
	if err != nil {
		return fmt.Errorf("error opening semaphore_ros_read: %w", err)
	}
	defer semaphoreROS.Close()

	// Memory-map the data file (read-only).
	s.dataMem, err = syscall.Mmap(int(dataFile.Fd()), 0, dataSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("error mapping shared_pc: %w", err)
	}

	// Memory-map the flag file with read/write permissions.
	s.flagReadMem, err = syscall.Mmap(int(flagRead.Fd()), 0, flagSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("error mapping semaphore_read: %w", err)
	}

	// Map the same file again for semaphore_ros_read (if intended to use a separate region, adjust accordingly).
	s.semaphore_ros_read, err = syscall.Mmap(int(semaphoreROS.Fd()), 0, flagSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("error mapping semaphore_ros_read: %w", err)
	}

	// Open or create the control shared memory file.
	controlFile, err := os.OpenFile("/dev/shm/shared_control", os.O_RDWR|os.O_CREATE, 0600)
	if err != nil {
		return fmt.Errorf("error opening shared_control: %w", err)
	}
	defer controlFile.Close()

	// Ensure the control file is the correct size.
	if err := controlFile.Truncate(controlSize); err != nil {
		return fmt.Errorf("error truncating shared_control: %w", err)
	}

	// Memory-map the control file with read/write permissions.
	s.controlMem, err = syscall.Mmap(int(controlFile.Fd()), 0, controlSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("error mapping shared_control: %w", err)
	}

	return nil
}

// getFrame reads a frame from shared memory in a synchronized manner.
// It waits until the flag indicates data is available, validates the frame length,
// and then returns the frame data. It also sets target bitrate and squid flag.
func (s *SharedMemory) getFrame(targetBitrate int, squid bool) ([]byte, error) {
	if s == nil {
		return nil, fmt.Errorf("shared memory not initialized")
	}

	for {
		// Check if data is ready, indicated by flagReadMem[0] being 1.
		if s.flagReadMem[0] == 1 {
			// Extract frame header and compute data length.
			header := s.dataMem[:headerSize]
			dataLength := uint32(header[0]) | uint32(header[1])<<8 | uint32(header[2])<<16

			// Validate the frame size against maximum allowed size.
			if dataLength > dataMaxSize {
				s.flagReadMem[0] = 0
				return nil, fmt.Errorf("invalid frame dimension: %d", dataLength)
			}

			// Copy frame data from the shared memory.
			data := make([]byte, dataLength)
			copy(data, s.dataMem[headerSize:headerSize+dataLength])

			// Save the target bitrate into flag memory (using bytes 2 to 6).
			binary.LittleEndian.PutUint32(s.flagReadMem[2:6], uint32(targetBitrate))

			// Set the squid flag accordingly.
			if squid {
				s.flagReadMem[1] = 1
			} else {
				s.flagReadMem[1] = 0
			}
			// Reset the flag to indicate the frame has been read.
			s.flagReadMem[0] = 0
			return data, nil
		}
		time.Sleep(1 * time.Millisecond)
	}
}

// Vector3 defines a three-dimensional vector.
type Vector3 struct {
	X float64
	Y float64
	Z float64
}

// writeControl writes control data (a counter and two Vector3 values) to shared memory.
// It uses a semaphore to ensure atomicity and validates that the data size matches controlSize.
func (s *SharedMemory) writeControl(counter int32, vec1 Vector3, vec2 Vector3) error {
	// Signal the start of the write operation.
	s.semaphore_ros_read[0] = 1

	// Prepare a buffer to accumulate all data.
	buffer := new(bytes.Buffer)

	// Write counter value.
	if err := binary.Write(buffer, binary.LittleEndian, counter); err != nil {
		return fmt.Errorf("error writing counter: %w", err)
	}

	// Write first Vector3 components.
	if err := binary.Write(buffer, binary.LittleEndian, vec1.X); err != nil {
		return fmt.Errorf("error writing vec1.X: %w", err)
	}
	if err := binary.Write(buffer, binary.LittleEndian, vec1.Y); err != nil {
		return fmt.Errorf("error writing vec1.Y: %w", err)
	}
	if err := binary.Write(buffer, binary.LittleEndian, vec1.Z); err != nil {
		return fmt.Errorf("error writing vec1.Z: %w", err)
	}

	// Write second Vector3 components.
	if err := binary.Write(buffer, binary.LittleEndian, vec2.X); err != nil {
		return fmt.Errorf("error writing vec2.X: %w", err)
	}
	if err := binary.Write(buffer, binary.LittleEndian, vec2.Y); err != nil {
		return fmt.Errorf("error writing vec2.Y: %w", err)
	}
	if err := binary.Write(buffer, binary.LittleEndian, vec2.Z); err != nil {
		return fmt.Errorf("error writing vec2.Z: %w", err)
	}

	// Get the complete byte slice.
	data := buffer.Bytes()

	// Validate data length.
	if len(data) != controlSize {
		return fmt.Errorf("unexpected data size: %d, expected: %d", len(data), controlSize)
	}

	// Copy the data into the mapped control memory.
	copy(s.controlMem, data)

	// Signal the end of the write operation.
	s.semaphore_ros_read[0] = 0

	return nil
}
