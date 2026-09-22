package cli

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestForcedDownloadPreservesPermissionsAndReplacesAtomically(t *testing.T) {
	for _, mode := range []os.FileMode{0o600, 0o640, 0o644} {
		t.Run(fmt.Sprintf("%04o", mode), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "artifact")
			if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(path, mode); err != nil {
				t.Fatal(err)
			}
			original, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer original.Close()
			if err := writeDownloadedFile(path, []byte("replacement"), true); err != nil {
				t.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if info.Mode().Perm() != mode {
				t.Errorf("replacement permissions = %04o, want %04o", info.Mode().Perm(), mode)
			}
			payload, err := os.ReadFile(path)
			if err != nil || string(payload) != "replacement" {
				t.Fatalf("replacement = %q, error %v", payload, err)
			}
			retained, err := io.ReadAll(original)
			if err != nil || string(retained) != "original" {
				t.Fatalf("previously opened file changed: %q, error %v", retained, err)
			}
			assertNoDownloadTemporaries(t, filepath.Dir(path))
		})
	}
}

func TestForcedDownloadNewFileIsPrivate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact")
	if err := writeDownloadedFile(path, []byte("private payload"), true); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o177 != 0 {
		t.Fatalf("new forced download grants permissions beyond 0600: %04o", info.Mode().Perm())
	}
}

func TestDownloadNewFileIsPrivate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact")
	if err := writeDownloadedFile(path, []byte("private payload"), false); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o177 != 0 {
		t.Fatalf("new download grants permissions beyond 0600: %04o", info.Mode().Perm())
	}
}

func TestDownloadRefusesExistingWithoutForce(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact")
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := writeDownloadedFile(path, []byte("replacement"), false); err == nil {
		t.Fatal("existing destination overwritten without --force")
	}
	data, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(data, []byte("original")) {
		t.Fatalf("original file changed: %q, %v", data, err)
	}
}

func TestForcedDownloadRejectsNonRegularDestinations(t *testing.T) {
	for _, kind := range []string{"directory", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "artifact")
			target := filepath.Join(root, "target")
			if err := os.WriteFile(target, []byte("untouched"), 0o600); err != nil {
				t.Fatal(err)
			}
			var err error
			if kind == "directory" {
				err = os.Mkdir(path, 0o700)
			} else {
				err = os.Symlink(target, path)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := writeDownloadedFile(path, []byte("replacement"), true); err == nil {
				t.Fatal("non-regular destination accepted")
			}
			data, err := os.ReadFile(target)
			if err != nil || string(data) != "untouched" {
				t.Fatalf("symlink target changed: %q, %v", data, err)
			}
			assertNoDownloadTemporaries(t, root)
		})
	}
}

func assertNoDownloadTemporaries(t *testing.T, directory string) {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(directory, ".contractor-download-*"))
	if err != nil || len(paths) != 0 {
		t.Fatalf("temporary downloads remain: %v, error %v", paths, err)
	}
}
