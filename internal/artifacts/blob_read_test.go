package artifacts

import (
	"bytes"
	"context"
	"errors"
	"io"
	"syscall"
	"testing"
)

// failingReader yields its data and then fails every read with err.
type failingReader struct {
	data []byte
	err  error
}

func (r *failingReader) Read(buffer []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, r.err
	}
	n := copy(buffer, r.data)
	r.data = r.data[n:]
	return n, nil
}

func TestReadBlobContentSeparatesStorageIOFromCorruption(t *testing.T) {
	content := bytes.Repeat([]byte("blob"), 40_000)
	for _, test := range []struct {
		name          string
		reader        io.Reader
		wantIO        bool
		wantIntegrity bool
	}{
		{name: "exact", reader: bytes.NewReader(content)},
		{name: "transient read failure", reader: &failingReader{data: content[:70_000], err: syscall.EIO}, wantIO: true},
		{name: "failure after the content", reader: &failingReader{data: content, err: syscall.EIO}, wantIO: true},
		{name: "short content", reader: bytes.NewReader(content[:len(content)-1]), wantIntegrity: true},
		{name: "long content", reader: bytes.NewReader(append(append([]byte{}, content...), 'x')), wantIntegrity: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			data, err := readBlobContent(context.Background(), test.reader, int64(len(content)))
			if errors.Is(err, ErrBlobIO) != test.wantIO || errors.Is(err, ErrArtifactIntegrity) != test.wantIntegrity {
				t.Fatalf("read error = %v, want I/O=%t integrity=%t", err, test.wantIO, test.wantIntegrity)
			}
			if err == nil && !bytes.Equal(data, content) {
				t.Fatal("exact read returned different content")
			}
		})
	}
}
