package securefile

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
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

func TestReadBeneathReadsRegularFilesWithinTheBound(t *testing.T) {
	root := t.TempDir()
	writeBeneath(t, root, "nested/dir/file.yaml", "abcd")
	writeBeneath(t, root, "empty.md", "")
	if data, err := ReadBeneath(root, "nested/dir/file.yaml", 4); err != nil || string(data) != "abcd" {
		t.Fatalf("ReadBeneath(at bound) = %q, %v", data, err)
	}
	if data, err := ReadBeneath(root, "empty.md", 4); err != nil || len(data) != 0 {
		t.Fatalf("ReadBeneath(empty) = %q, %v", data, err)
	}
	if data, err := ReadBeneath(root, "nested/dir/file.yaml", 3); !errors.Is(err, ErrTooLarge) || data != nil {
		t.Fatalf("ReadBeneath(over bound) = %q, %v", data, err)
	}
}

// Every component is opened without following a symlink, so a link anywhere
// on the path, including the root itself, is refused rather than followed.
func TestReadBeneathRefusesSymlinksAndOtherFileTypes(t *testing.T) {
	root := t.TempDir()
	writeBeneath(t, root, "real/file.yaml", "value")
	outside := writeFile(t, "outside", "secret", 0o600)
	for link, target := range map[string]string{
		"link.yaml":    "real/file.yaml",
		"escape.yaml":  outside,
		"linked":       "real",
		"real/up.yaml": "../real/file.yaml",
	} {
		if err := os.Symlink(target, filepath.Join(root, link)); err != nil {
			t.Fatal(err)
		}
	}
	linkedRoot := filepath.Join(t.TempDir(), "root")
	if err := os.Symlink(root, linkedRoot); err != nil {
		t.Fatal(err)
	}
	if err := unix.Mkfifo(filepath.Join(root, "fifo"), 0o600); err != nil {
		t.Fatal(err)
	}
	for name, read := range map[string]func() ([]byte, error){
		"final symlink":        func() ([]byte, error) { return ReadBeneath(root, "link.yaml", 64) },
		"escaping symlink":     func() ([]byte, error) { return ReadBeneath(root, "escape.yaml", 64) },
		"directory symlink":    func() ([]byte, error) { return ReadBeneath(root, "linked/file.yaml", 64) },
		"relative symlink":     func() ([]byte, error) { return ReadBeneath(root, "real/up.yaml", 64) },
		"symlinked root":       func() ([]byte, error) { return ReadBeneath(linkedRoot, "real/file.yaml", 64) },
		"missing file":         func() ([]byte, error) { return ReadBeneath(root, "real/missing.yaml", 64) },
		"file as directory":    func() ([]byte, error) { return ReadBeneath(root, "real/file.yaml/child", 64) },
		"directory as file":    func() ([]byte, error) { return ReadBeneath(root, "real", 64) },
		"FIFO without writers": func() ([]byte, error) { return ReadBeneath(root, "fifo", 64) },
	} {
		t.Run(name, func(t *testing.T) {
			result := make(chan error, 1)
			go func() {
				data, err := read()
				if err == nil || data != nil || errors.Is(err, ErrTooLarge) {
					err = errors.Join(errors.New("unsafe path was read"), err)
				}
				result <- err
			}()
			select {
			case err := <-result:
				if strings.HasPrefix(err.Error(), "unsafe path was read") {
					t.Fatal(err)
				}
			case <-time.After(5 * time.Second):
				// Unblock a stuck FIFO open so the goroutine can finish.
				if writer, err := os.OpenFile(filepath.Join(root, "fifo"), os.O_WRONLY, 0); err == nil {
					_ = writer.Close()
				}
				t.Fatal("ReadBeneath blocked")
			}
		})
	}
	// A symlink or another file type is a changed path, not an
	// operating-system failure; a missing file keeps its own error.
	for name, read := range map[string]func() ([]byte, error){
		"final symlink":     func() ([]byte, error) { return ReadBeneath(root, "link.yaml", 64) },
		"directory symlink": func() ([]byte, error) { return ReadBeneath(root, "linked/file.yaml", 64) },
		"symlinked root":    func() ([]byte, error) { return ReadBeneath(linkedRoot, "real/file.yaml", 64) },
		"file as directory": func() ([]byte, error) { return ReadBeneath(root, "real/file.yaml/child", 64) },
		"directory":         func() ([]byte, error) { return ReadBeneath(root, "real", 64) },
		"FIFO":              func() ([]byte, error) { return ReadBeneath(root, "fifo", 64) },
	} {
		var errno unix.Errno
		if _, err := read(); !errors.Is(err, ErrNotRegular) || errors.As(err, &errno) {
			t.Errorf("ReadBeneath(%s) error = %v, want ErrNotRegular without an errno", name, err)
		}
	}
	if _, err := ReadBeneath(root, "real/missing.yaml", 64); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("ReadBeneath(missing) error = %v", err)
	}
}

func TestReadBeneathRejectsPathsThatLeaveTheRoot(t *testing.T) {
	root := t.TempDir()
	writeBeneath(t, root, "a/file", "value")
	for _, relative := range []string{"", "/a/file", "a//file", "./a/file", "a/./file", "a/../a/file", "..", "a/file/"} {
		if data, err := ReadBeneath(root, relative, 64); !errors.Is(err, ErrBeneathPath) || data != nil {
			t.Errorf("ReadBeneath(%q) = %q, %v", relative, data, err)
		}
	}
}

func writeBeneath(t *testing.T, root, relative, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
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
