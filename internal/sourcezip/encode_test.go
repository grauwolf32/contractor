package sourcezip

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"testing"
	"time"
)

func TestEncodeIsDeterministicAndEnforcesArchiveLimit(t *testing.T) {
	members := []Member{{
		Name: "nested/hello.txt", Size: 6,
		Write: func(w io.Writer) error {
			_, err := io.WriteString(w, "hello\n")
			return err
		},
	}}
	options := Options{CompressionLevel: 6}
	first, err := Encode(context.Background(), members, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Encode(context.Background(), members, options)
	if err != nil || !bytes.Equal(first, second) {
		t.Fatalf("source ZIP changed between builds: %v", err)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(first)); digest != "e88aa24a6b73f0be8879ed3a5822c6e1e99dde90e1aee1e8ed445535f40a60c3" {
		t.Fatalf("source ZIP bytes changed from the legacy writer: %s", digest)
	}
	reader, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil || len(reader.File) != 1 || reader.File[0].Name != members[0].Name ||
		reader.File[0].Mode().Perm() != 0o644 || reader.File[0].Modified.Year() != 1980 {
		t.Fatalf("source ZIP metadata = (%+v, %v)", reader, err)
	}
	options.MaxBytes = len(first) - 1
	if _, err := Encode(context.Background(), members, options); !errors.Is(err, ErrLimit) {
		t.Fatalf("compressed archive limit error = %v", err)
	}
	options.MaxBytes = MaxArchiveBytes + 1
	if _, err := Encode(context.Background(), members, options); !errors.Is(err, ErrLimit) {
		t.Fatalf("oversized archive allowance error = %v", err)
	}
	members[0].Size = MaxFileBytes + 1
	if _, err := Encode(context.Background(), members, Options{}); !errors.Is(err, ErrLimit) {
		t.Fatalf("expanded member limit error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	members[0].Size = 6
	if _, err := Encode(ctx, members, Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled archive error = %v", err)
	}
}

func TestBoundedWriterNeverExceedsMaximum(t *testing.T) {
	var buffer bytes.Buffer
	writer := &boundedWriter{ctx: context.Background(), buffer: &buffer, maximum: 4}
	if written, err := writer.Write([]byte("1234")); err != nil || written != 4 {
		t.Fatalf("exact write = %d, %v", written, err)
	}
	if written, err := writer.Write([]byte("5")); !errors.Is(err, ErrLimit) || written != 0 {
		t.Fatalf("overflow write = %d, %v", written, err)
	}
	if buffer.Len() != 4 {
		t.Fatalf("buffer length = %d", buffer.Len())
	}
}

func TestModifiedFieldHeaderKeepsGitImportTimestamp(t *testing.T) {
	archive, err := Encode(context.Background(), []Member{{
		Name: "file.txt", Size: 0, Write: func(io.Writer) error { return nil },
	}}, Options{ModifiedField: true})
	if err != nil {
		t.Fatal(err)
	}
	reader, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil || len(reader.File) != 1 ||
		!reader.File[0].Modified.Equal(time.Date(1980, 1, 1, 0, 0, 0, 0, time.UTC)) {
		t.Fatalf("Git import ZIP timestamp = (%+v, %v)", reader, err)
	}
}
