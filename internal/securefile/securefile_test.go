package securefile

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestReadAcceptsBoundedOwnerOnlyFile(t *testing.T) {
	path := writeFile(t, "secret", "value", 0o600)
	data, err := Read(path, 5)
	if err != nil || string(data) != "value" {
		t.Fatalf("Read() = %q, %v", data, err)
	}
}

func TestReadRejectsUnsafeFiles(t *testing.T) {
	directory := t.TempDir()
	safe := writeFile(t, "safe", "value", 0o600)
	link := filepath.Join(directory, "link")
	if err := os.Symlink(safe, link); err != nil {
		t.Fatal(err)
	}
	cases := map[string]struct {
		path     string
		maxBytes int64
		want     error
	}{
		"empty path":     {"", 16, ErrPath},
		"blank path":     {"  ", 16, ErrPath},
		"relative path":  {"secret", 16, ErrPath},
		"unclean path":   {directory + "/./safe", 16, ErrPath},
		"missing file":   {filepath.Join(directory, "missing"), 16, ErrUnsafePath},
		"symlink":        {link, 16, ErrUnsafePath},
		"directory":      {directory, 16, ErrUnsafePath},
		"group readable": {writeFile(t, "group", "value", 0o640), 16, ErrUnsafePath},
		"other readable": {writeFile(t, "other", "value", 0o604), 16, ErrUnsafePath},
		"empty file":     {writeFile(t, "empty", "", 0o600), 16, ErrUnsafeHandle},
		"oversized file": {writeFile(t, "large", "value", 0o600), 4, ErrUnsafeHandle},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			data, err := Read(tc.path, tc.maxBytes)
			if !errors.Is(err, tc.want) || data != nil {
				t.Fatalf("Read() = %q, %v; want %v", data, err, tc.want)
			}
		})
	}
}

func TestReadRestrictedAppliesForbiddenBits(t *testing.T) {
	path := writeFile(t, "bindings", "value", 0o640)
	if data, err := ReadRestricted(path, 16, 0o022); err != nil || string(data) != "value" {
		t.Fatalf("ReadRestricted(0o022) = %q, %v", data, err)
	}
	if _, err := ReadRestricted(writeFile(t, "writable", "value", 0o620), 16, 0o022); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("ReadRestricted(group-writable) error = %v", err)
	}
}

func TestReadRejectsFileOwnedByAnotherUser(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("changing a file owner requires root")
	}
	path := writeFile(t, "foreign", "value", 0o600)
	if err := os.Chown(path, 4242, 4242); err != nil {
		t.Fatal(err)
	}
	if data, err := Read(path, 16); !errors.Is(err, ErrUnsafePath) || data != nil {
		t.Fatalf("Read(foreign owner) = %q, %v", data, err)
	}
	if data, err := readOpened(path, 16, OwnerOnly); !errors.Is(err, ErrUnsafeHandle) || data != nil {
		t.Fatalf("readOpened(foreign owner) = %q, %v", data, err)
	}
}

// A FIFO swapped in after the path check must be refused rather than block
// startup until some writer opens it.
func TestReadOpenedRefusesFIFOWithoutBlocking(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fifo")
	if err := unix.Mkfifo(path, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(path, 16); !errors.Is(err, ErrUnsafePath) {
		t.Fatalf("Read(FIFO) error = %v", err)
	}
	result := make(chan error, 1)
	go func() {
		_, err := readOpened(path, 16, OwnerOnly)
		result <- err
	}()
	select {
	case err := <-result:
		if !errors.Is(err, ErrUnsafeHandle) {
			t.Fatalf("readOpened(FIFO) error = %v", err)
		}
	case <-time.After(5 * time.Second):
		// Unblock the stuck open so the goroutine can finish.
		if writer, err := os.OpenFile(path, os.O_WRONLY, 0); err == nil {
			_ = writer.Close()
		}
		t.Fatal("readOpened blocked on a FIFO")
	}
}

func writeFile(t *testing.T, name, content string, mode os.FileMode) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, mode); err != nil {
		t.Fatal(err)
	}
	return path
}
