package artifacts

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFilesystemBlobBoundaryAndGenerations(t *testing.T) {
	ctx := context.Background()
	s, err := OpenFilesystemBlobStore(ctx, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, size := range []int{0, 64 << 20} {
		data := make([]byte, size, size+1)
		if size > 0 {
			data[0], data[size-1] = 1, 255
		}
		first, err := s.Store(ctx, data)
		if err != nil {
			t.Fatal(err)
		}
		second, err := s.Store(ctx, data)
		if err != nil {
			t.Fatal(err)
		}
		if first.Key == second.Key {
			t.Fatal("physical key reused")
		}
		if err := s.Delete(ctx, first); err != nil {
			t.Fatal(err)
		}
		read, err := s.Read(ctx, second)
		if err != nil || !bytes.Equal(read, data) {
			t.Fatalf("read: %v", err)
		}
		if _, err := s.Read(ctx, first); !errors.Is(err, ErrBlobMissing) {
			t.Fatalf("missing: %v", err)
		}
		if size == 64<<20 {
			if _, err := s.Store(ctx, append(data, 0)); !errors.Is(err, ErrPayloadTooLarge) {
				t.Fatalf("oversize: %v", err)
			}
		}
	}
}

func TestFilesystemBlobRejectsEscapesCorruptionAndCancellation(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir()
	s, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	object, err := s.Store(ctx, []byte("original"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, object.Key), []byte("modified"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Read(ctx, object); !errors.Is(err, ErrArtifactIntegrity) {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(outside, []byte("outside"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(path, object.Key)); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(path, object.Key)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Read(ctx, object); err == nil {
		t.Fatal("symlink read")
	}
	if err := s.Delete(ctx, object); err == nil {
		t.Fatal("symlink delete")
	}
	object.Key = "../outside"
	if _, err := s.Read(ctx, object); err == nil {
		t.Fatal("escape read")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := s.Store(cancelled, []byte("x")); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".staging-") {
			t.Fatal("staging leaked after successful or cancelled store")
		}
	}
	link := filepath.Join(t.TempDir(), "root-link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenFilesystemBlobStore(ctx, link); err == nil {
		t.Fatal("symlink root accepted")
	}
}

func TestFilesystemBlobUnwritableRoot(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses mode permissions")
	}
	path := t.TempDir()
	if err := os.Chmod(path, 0500); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(path, 0700)
	if _, err := OpenFilesystemBlobStore(context.Background(), path); err == nil {
		t.Fatal("unwritable root accepted")
	}
}
