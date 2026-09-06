package sourcebundle

import (
	"archive/zip"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func TestLimitedBufferNeverExceedsMaximum(t *testing.T) {
	buffer := &limitedBuffer{maximum: 4}
	if written, err := buffer.Write([]byte("1234")); err != nil || written != 4 {
		t.Fatalf("exact write = %d, %v", written, err)
	}
	if written, err := buffer.Write([]byte("5")); err != ErrArchiveTooLarge || written != 0 {
		t.Fatalf("overflow write = %d, %v", written, err)
	}
	if buffer.Len() != 4 {
		t.Fatalf("buffer length = %d", buffer.Len())
	}
}

func TestBuildIsDeterministicAndUsesDirectoryContentsAsRoot(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "z.txt"), "z")
	writeTestFile(t, filepath.Join(root, "nested", "a.txt"), "a")
	first, err := Build(root, Options{IncludeIgnored: true})
	if err != nil {
		t.Fatal(err)
	}
	second, err := Build(root, Options{IncludeIgnored: true})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Data, second.Data) || first.SHA256 != second.SHA256 {
		t.Fatal("same working tree produced different source ZIP bytes")
	}
	reader, err := zip.NewReader(bytes.NewReader(first.Data), int64(len(first.Data)))
	if err != nil {
		t.Fatal(err)
	}
	paths := make([]string, 0, len(reader.File))
	for _, file := range reader.File {
		paths = append(paths, file.Name)
	}
	sort.Strings(paths)
	if expected := []string{"nested/a.txt", "z.txt"}; !reflect.DeepEqual(paths, expected) {
		t.Fatalf("paths = %v, want %v", paths, expected)
	}
}

func TestBuildRejectsSymlink(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "target"), "target")
	if err := os.Symlink("target", filepath.Join(root, "link")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if _, err := Build(root, Options{IncludeIgnored: true}); err == nil {
		t.Fatal("symlink was accepted")
	}
}

func TestBuildHonorsGitAndContractorIgnore(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is unavailable")
	}
	root := t.TempDir()
	if output, err := exec.Command("git", "-C", root, "init", "--quiet").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, output)
	}
	writeTestFile(t, filepath.Join(root, ".gitignore"), "ignored.txt\n")
	writeTestFile(t, filepath.Join(root, ".contractorignore"), "private.txt\n")
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	writeTestFile(t, filepath.Join(root, "ignored.txt"), "ignored")
	writeTestFile(t, filepath.Join(root, "private.txt"), "private")
	writeTestFile(t, filepath.Join(root, ".contractor", "local-state"), "secret")
	bundle, err := Build(root, Options{})
	if err != nil {
		t.Fatal(err)
	}
	reader, err := zip.NewReader(bytes.NewReader(bundle.Data), int64(len(bundle.Data)))
	if err != nil {
		t.Fatal(err)
	}
	paths := make([]string, 0, len(reader.File))
	for _, file := range reader.File {
		paths = append(paths, file.Name)
	}
	sort.Strings(paths)
	expected := []string{".contractorignore", ".gitignore", "kept.go"}
	if !reflect.DeepEqual(paths, expected) {
		t.Fatalf("paths = %v, want %v", paths, expected)
	}
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
