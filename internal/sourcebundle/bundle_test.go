package sourcebundle

import (
	"archive/zip"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
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

func TestBuildHonorsContractorIgnoreFromAnotherWorkingDirectory(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, ".contractorignore"), "/private.txt\n")
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	writeTestFile(t, filepath.Join(root, "private.txt"), "private")
	writeTestFile(t, filepath.Join(root, "nested", "private.txt"), "nested")
	t.Chdir(filepath.Join(root, "nested"))
	assertBundlePaths(t, root, []string{".contractorignore", "kept.go", "nested/private.txt"})
}

func TestBuildKeepsContractorIgnoreNegations(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, ".contractorignore"), "*.log\n!keep.log\n")
	writeTestFile(t, filepath.Join(root, "drop.log"), "drop")
	writeTestFile(t, filepath.Join(root, "keep.log"), "keep")
	assertBundlePaths(t, root, []string{".contractorignore", "keep.log"})
}

func TestBuildSkipsSubmodulesAndNestedRepositories(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	if err := os.MkdirAll(filepath.Join(root, "module"), 0o700); err != nil {
		t.Fatal(err)
	}
	gitlink := "160000,0123456789abcdef0123456789abcdef01234567,module"
	if output, err := exec.Command("git", "-C", root, "update-index", "--add", "--cacheinfo", gitlink).CombinedOutput(); err != nil {
		t.Fatalf("git update-index: %v: %s", err, output)
	}
	nested := filepath.Join(root, "nested")
	if output, err := exec.Command("git", "init", "--quiet", nested).CombinedOutput(); err != nil {
		t.Fatalf("git init nested: %v: %s", err, output)
	}
	writeTestFile(t, filepath.Join(nested, "inner.go"), "package inner")
	bundle := assertBundlePaths(t, root, []string{"kept.go"})
	if bundle.SkippedRepositories != 2 {
		t.Fatalf("skipped repositories = %d, want 2", bundle.SkippedRepositories)
	}
}

func TestBuildReportsGitFailureInsideWorkTree(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	writeTestFile(t, filepath.Join(root, ".git", "index"), "corrupt")
	_, err := Build(root, Options{})
	if err == nil || !strings.Contains(err.Error(), "--include-ignored") || !strings.Contains(err.Error(), "index") {
		t.Fatalf("Build error = %v, want Git stderr and --include-ignored hint", err)
	}
	if _, err := Build(root, Options{IncludeIgnored: true}); err != nil {
		t.Fatalf("Build with IncludeIgnored: %v", err)
	}
}

func TestBuildWalksPlainDirectoryWithoutGit(t *testing.T) {
	root := t.TempDir()
	if insideGitWorkTree(root) {
		t.Skip("temporary directory is inside a Git work tree")
	}
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	assertBundlePaths(t, root, []string{"kept.go"})
}

func TestBuildEnforcesSourceLimits(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "large.bin"), strings.Repeat("x", MaxFileBytes+1))
	if _, err := Build(root, Options{IncludeIgnored: true}); err == nil || !strings.Contains(err.Error(), "per-file limit") {
		t.Fatalf("oversized file error = %v", err)
	}

	root = t.TempDir()
	long := filepath.Join(strings.Repeat("d", 200), strings.Repeat("e", 200), strings.Repeat("f", 200))
	writeTestFile(t, filepath.Join(root, long), "x")
	if _, err := Build(root, Options{IncludeIgnored: true}); err == nil || !strings.Contains(err.Error(), "path limit") {
		t.Fatalf("long path error = %v", err)
	}

	root = t.TempDir()
	for index := 0; index <= MaxEntries; index++ {
		writeTestFile(t, filepath.Join(root, strconv.Itoa(index%100), strconv.Itoa(index)), "")
	}
	if _, err := Build(root, Options{IncludeIgnored: true}); err == nil || !strings.Contains(err.Error(), "file limit") {
		t.Fatalf("entry count error = %v", err)
	}
}

func initGitRepository(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is unavailable")
	}
	root := t.TempDir()
	if output, err := exec.Command("git", "-C", root, "init", "--quiet").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, output)
	}
	return root
}

func assertBundlePaths(t *testing.T, root string, expected []string) Bundle {
	t.Helper()
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
	if !reflect.DeepEqual(paths, expected) {
		t.Fatalf("paths = %v, want %v", paths, expected)
	}
	return bundle
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
