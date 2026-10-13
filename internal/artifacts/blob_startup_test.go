package artifacts

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestAutoBlobFallbackWithoutDirectoryAndAfterFileFailure(t *testing.T) {
	data := bytes.Repeat([]byte{0x77}, 1048577)
	for _, configured := range []bool{false, true} {
		t.Run(map[bool]string{false: "no-directory", true: "file-failure"}[configured], func(t *testing.T) {
			path := ""
			if configured {
				path = filepath.Join(t.TempDir(), "blobs")
			}
			_, store, err := OpenBlobStore(t.Context(), BlobAuto, path, DefaultBlobInlineMaxBytes)
			if err != nil {
				t.Fatal(err)
			}
			defer store.(io.Closer).Close()
			if configured {
				// Remove the initialized volume before preparing another upload.
				if err := os.RemoveAll(path); err != nil {
					t.Fatal(err)
				}
			}
			object, err := store.Store(t.Context(), data)
			if err != nil || object.Backend != BlobPostgres {
				t.Fatalf("fallback object: %s, %v", object.Backend, err)
			}
			read, err := store.Read(t.Context(), object)
			if err != nil || !bytes.Equal(read, data) {
				t.Fatalf("fallback bytes: %v", err)
			}
		})
	}
}

func TestAutoBlobReadOnlyFilesystemStillReadsExistingFiles(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permission test requires a user without root filesystem privileges")
	}
	path := t.TempDir()
	files, err := OpenFilesystemBlobStore(t.Context(), path)
	if err != nil {
		t.Fatal(err)
	}
	object, err := files.Store(t.Context(), []byte("retained file"))
	files.Close()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0500); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(path, 0700)
	backend, store, err := OpenBlobStore(t.Context(), BlobAuto, path, DefaultBlobInlineMaxBytes)
	if err != nil || backend != BlobPostgres {
		t.Fatalf("read-only root: %s, %v", backend, err)
	}
	defer store.(io.Closer).Close()
	read, err := store.Read(t.Context(), object)
	if err != nil || string(read) != "retained file" {
		t.Fatalf("readable existing file: %v", err)
	}
	newObject, err := store.Store(t.Context(), bytes.Repeat([]byte{0x55}, 1048577))
	if err != nil || newObject.Backend != BlobPostgres {
		t.Fatalf("read-only fallback: %s, %v", newObject.Backend, err)
	}
}

func TestBlobStorageCancelledStartupDoesNotProbeOrFallback(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		var ctx context.Context
		var cancel context.CancelFunc
		want := error(context.Canceled)
		if deadline {
			ctx, cancel = context.WithDeadline(t.Context(), time.Time{})
			want = context.DeadlineExceeded
		} else {
			ctx, cancel = context.WithCancel(t.Context())
			cancel()
		}
		defer cancel()
		path := filepath.Join(t.TempDir(), "blobs")
		_, store, err := OpenBlobStore(ctx, BlobAuto, path, DefaultBlobInlineMaxBytes)
		if !errors.Is(err, want) || store != nil {
			t.Fatalf("cancelled startup: %v", err)
		}
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("cancelled startup created root: %v", err)
		}
	}
}

func TestBlobInlineZeroAndMaximumThresholds(t *testing.T) {
	for _, threshold := range []int64{0, MaxPayloadSize} {
		_, store, err := OpenBlobStore(t.Context(), BlobAuto, t.TempDir(), threshold)
		if err != nil {
			t.Fatal(err)
		}
		defer store.(io.Closer).Close()
		for _, data := range [][]byte{nil, {1}} {
			object, err := store.Store(t.Context(), data)
			want := BlobFilesystem
			if int64(len(data)) <= threshold {
				want = BlobPostgres
			}
			if err != nil || object.Backend != want {
				t.Fatalf("threshold %d, size %d: %s, %v", threshold, len(data), object.Backend, err)
			}
		}
	}
}
