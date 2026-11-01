package zipdirectory

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
)

func archive(t *testing.T, entries int) []byte {
	t.Helper()
	var data bytes.Buffer
	writer := zip.NewWriter(&data)
	for range entries {
		if _, err := writer.CreateHeader(&zip.FileHeader{Name: "a", Method: zip.Store}); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return data.Bytes()
}

func TestCheckCountsActualRecordsAgainstTheCallerLimit(t *testing.T) {
	data := archive(t, 3)
	start, err := Check(data, 3)
	if err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint32(data[start:]) != 0x02014b50 {
		t.Fatalf("offset %d is not the first directory record", start)
	}
	if _, err := Check(data, 2); !errors.Is(err, ErrLimit) {
		t.Fatalf("over the caller limit = %v", err)
	}
	// Lie about both record counts: only the actual records count.
	binary.LittleEndian.PutUint16(data[len(data)-14:], 1)
	binary.LittleEndian.PutUint16(data[len(data)-12:], 1)
	if _, err := Check(data, 2); !errors.Is(err, ErrLimit) {
		t.Fatalf("understated count over the limit = %v", err)
	}
	if _, err := Check(data, 3); !errors.Is(err, ErrInvalid) {
		t.Fatalf("understated count within the limit = %v", err)
	}
	for _, broken := range [][]byte{nil, []byte("PK"), append(archive(t, 1), 0)} {
		if _, err := Check(broken, 10); !errors.Is(err, ErrInvalid) {
			t.Fatalf("broken archive %x = %v", broken, err)
		}
	}
}
