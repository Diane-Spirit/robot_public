package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

// ManagedCSVFile encapsulates the management of a CSV file, ensuring header consistency
// and providing a simple interface for appending data.
type ManagedCSVFile struct {
	filePath string   // Path to the CSV file.
	header   []string // Expected header for the CSV file.
}

// slicesEqual checks whether two string slices are identical.
func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

// overwriteFileWithHeader truncates and/or creates the CSV file at filePath and writes the header.
// This ensures that the file starts with the correct header.
func overwriteFileWithHeader(filePath string, header []string) error {
	// Open the file with O_TRUNC to clear its content, or create it if it doesn't exist.
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		log.Printf("Error opening file %s for overwrite: %v", filePath, err)
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.Write(header); err != nil {
		log.Printf("Error writing header to CSV file %s: %v", filePath, err)
		return err
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		log.Printf("Error flushing CSV writer for %s: %v", filePath, err)
		return err
	}
	log.Printf("Header successfully written to %s.", filePath)
	return nil
}

// NewManagedCSV creates a ManagedCSVFile instance at filePath with the specified header.
// If forceRecreate is true, an existing file will be overwritten with the header.
// Otherwise, the function will check the existing file header and overwrite if necessary.
func NewManagedCSV(filePath string, header []string, forceRecreate bool) (*ManagedCSVFile, error) {
	if forceRecreate {
		log.Printf("Forcing recreation of CSV file %s with new header.", filePath)
		if err := overwriteFileWithHeader(filePath, header); err != nil {
			return nil, err
		}
	} else {
		// Attempt to open for read/write to verify the existing header; create the file if missing.
		file, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, 0644)
		if err != nil {
			log.Printf("Error opening/creating CSV file %s for header check: %v", filePath, err)
			return nil, err
		}

		reader := csv.NewReader(file)
		existingHeader, readErr := reader.Read()

		needsOverwrite := false
		if readErr == io.EOF { // File is empty or newly created.
			log.Printf("File %s is new or empty. Writing header.", filePath)
			needsOverwrite = true
		} else if readErr != nil { // Error reading header (e.g. malformed CSV).
			log.Printf("Error reading header from CSV file %s (error: %v). Overwriting.", filePath, readErr)
			needsOverwrite = true
		} else if !slicesEqual(existingHeader, header) { // Header mismatch.
			log.Printf("Header in %s is incorrect. Overwriting with new header.", filePath)
			needsOverwrite = true
		}

		if needsOverwrite {
			file.Close() // Close the file before overwriting.
			if err := overwriteFileWithHeader(filePath, header); err != nil {
				return nil, err
			}
		} else {
			log.Printf("File %s already exists with the correct header.", filePath)
			file.Close()
		}
	}

	return &ManagedCSVFile{
		filePath: filePath,
		header:   header,
	}, nil
}

// AppendData appends a single data row to the CSV file.
// It opens the file in append mode, writes the row, flushes and checks for errors.
func (m *ManagedCSVFile) AppendData(dataRow []string) error {
	// Open the file in append mode with O_CREATE as a safeguard.
	file, err := os.OpenFile(m.filePath, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Printf("Error opening CSV file %s in append mode: %v", m.filePath, err)
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.Write(dataRow); err != nil {
		log.Printf("Error writing record to CSV %s: %v", m.filePath, err)
		return err
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		log.Printf("Error after flushing CSV writer for file %s: %v", m.filePath, err)
		return err
	}

	return nil
}

// AddDataRecord is an example exportable function to record data into the CSV file.
// It converts specific parameters to a string slice and appends the record.
// For more complex data structures, consider JSON decoding/encoding as needed.
func AddDataRecord(filePath string, webSocketTime string, nopReceived int64, nopOriginal int64, lossRate float64) error {
	// Define the expected header for this data set.
	header := []string{"WebSocketTime", "nopReceived", "nopOriginal", "lossRate"}

	// Initialize the managed CSV file without forcing recreation.
	csvFile, err := NewManagedCSV(filePath, header, false)
	if err != nil {
		log.Printf("Error initializing managed CSV for AddDataRecord: %v", err)
		return err
	}

	record := []string{
		webSocketTime,
		strconv.FormatInt(nopReceived, 10),
		strconv.FormatInt(nopOriginal, 10),
		strconv.FormatFloat(lossRate, 'f', 4, 64),
	}
	return csvFile.AppendData(record)
}
