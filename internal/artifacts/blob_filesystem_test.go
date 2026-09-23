package artifacts

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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

func TestFilesystemBlobFlushesBeforePublishing(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir()
	s, err := OpenFilesystemBlobStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	// Drop the probe's empty shard so the next Store must create its shard.
	shards, err := os.ReadDir(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, shard := range shards {
		if err := os.Remove(filepath.Join(path, shard.Name())); err != nil {
			t.Fatal(err)
		}
	}
	var flushed []string
	s.sync = func(f *os.File) error {
		info, err := f.Stat()
		if err != nil {
			return err
		}
		name := filepath.Base(f.Name())
		if info.Mode().IsRegular() {
			published, err := filepath.Glob(filepath.Join(path, "*", strings.TrimPrefix(name, ".staging-")))
			if err != nil || len(published) != 0 || info.Size() != int64(len("durable")) {
				t.Errorf("file flush %s saw published=%v size=%d", name, published, info.Size())
			}
			name = "file"
		}
		flushed = append(flushed, name)
		return f.Sync()
	}
	object, err := s.Store(ctx, []byte("durable"))
	if err != nil {
		t.Fatal(err)
	}
	if expected := []string{".", "file", object.Key[:2]}; strings.Join(flushed, ",") != strings.Join(expected, ",") {
		t.Fatalf("flushes = %v, want %v", flushed, expected)
	}

	for shard := range 256 {
		if err := os.Mkdir(filepath.Join(path, fmt.Sprintf("%02x", shard)), 0o700); err != nil && !errors.Is(err, os.ErrExist) {
			t.Fatal(err)
		}
	}
	flushed = nil
	object, err = s.Store(ctx, []byte("durable"))
	if err != nil {
		t.Fatal(err)
	}
	if expected := []string{"file", object.Key[:2]}; strings.Join(flushed, ",") != strings.Join(expected, ",") {
		t.Fatalf("flushes into an existing shard = %v, want %v", flushed, expected)
	}

	s.sync = func(f *os.File) error {
		if info, err := f.Stat(); err == nil && info.IsDir() {
			return errors.New("injected directory flush failure")
		}
		return nil
	}
	if _, err := s.Store(ctx, []byte("durable")); err == nil {
		t.Fatal("Store succeeded after a failed directory flush")
	}
	files := 0
	err = filepath.WalkDir(path, func(_ string, entry os.DirEntry, err error) error {
		if err == nil && !entry.IsDir() {
			files++
		}
		return err
	})
	if err != nil || files != 2 {
		t.Fatalf("files after failed flush = %d, %v; want the two published generations", files, err)
	}
}
