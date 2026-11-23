package sourcebundle

import (
	"archive/zip"
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestBuildPackagesUnmergedIndexPathOnce(t *testing.T) {
	for _, kind := range []string{"content", "modify/delete"} {
		t.Run(kind, func(t *testing.T) {
			root := conflictedRepository(t, kind)
			output := gitOutput(t, root, "ls-files", "--cached", "-z")
			stages := 0
			for _, path := range splitNUL(output) {
				if path == "conflict.txt" {
					stages++
				}
			}
			if stages < 2 {
				t.Fatalf("test fixture has %d index stages, want at least 2", stages)
			}

			bundle, err := Build(root, Options{})
			if err != nil {
				t.Fatalf("Build conflicted working tree: %v", err)
			}
			if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, []string{"conflict.txt"}) {
				t.Fatalf("bundle paths = %v", paths)
			}
			want, err := os.ReadFile(filepath.Join(root, "conflict.txt"))
			if err != nil {
				t.Fatal(err)
			}
			if got := bundledFile(t, bundle, "conflict.txt"); !bytes.Equal(got, want) {
				t.Fatalf("bundle has %q, want working-tree bytes %q", got, want)
			}
		})
	}
}

func TestBuildStillRejectsDistinctUnicodeNamesThatNormalizeTogether(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, "café.txt"), "NFC")
	writeTestFile(t, filepath.Join(root, "cafe\u0301.txt"), "NFD")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 { // Two files plus .git; some filesystems normalize names.
		t.Skip("filesystem does not preserve distinct NFC and NFD names")
	}
	if _, err := Build(root, Options{}); err == nil || !strings.Contains(err.Error(), "paths collide after normalization") {
		t.Fatalf("distinct raw names with the same NFC path = %v", err)
	}
}

func conflictedRepository(t *testing.T, kind string) string {
	t.Helper()
	root := initGitRepository(t)
	gitOutput(t, root, "config", "user.name", "Source Bundle Test")
	gitOutput(t, root, "config", "user.email", "source-bundle@example.test")
	path := filepath.Join(root, "conflict.txt")
	writeTestFile(t, path, "base\n")
	gitAdd(t, root, "conflict.txt")
	gitOutput(t, root, "commit", "-m", "base")
	mainBranch := strings.TrimSpace(string(gitOutput(t, root, "branch", "--show-current")))
	gitOutput(t, root, "checkout", "-b", "side")
	if kind == "modify/delete" {
		gitOutput(t, root, "rm", "conflict.txt")
	} else {
		writeTestFile(t, path, "side\n")
		gitAdd(t, root, "conflict.txt")
	}
	gitOutput(t, root, "commit", "-m", "side")
	gitOutput(t, root, "checkout", mainBranch)
	writeTestFile(t, path, "main\n")
	gitAdd(t, root, "conflict.txt")
	gitOutput(t, root, "commit", "-m", "main")
	output, err := exec.Command("git", "-C", root, "merge", "side").CombinedOutput()
	if err == nil || !bytes.Contains(output, []byte("CONFLICT")) {
		t.Fatalf("git merge did not create %s conflict: %v: %s", kind, err, output)
	}
	return root
}

func gitOutput(t *testing.T, root string, args ...string) []byte {
	t.Helper()
	command := exec.Command("git", append([]string{"-C", root}, args...)...)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, output)
	}
	return output
}

func bundledFile(t *testing.T, bundle Bundle, name string) []byte {
	t.Helper()
	reader, err := zip.NewReader(bytes.NewReader(bundle.Data), int64(len(bundle.Data)))
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range reader.File {
		if file.Name != name {
			continue
		}
		opened, err := file.Open()
		if err != nil {
			t.Fatal(err)
		}
		defer opened.Close()
		content, err := io.ReadAll(opened)
		if err != nil {
			t.Fatal(err)
		}
		return content
	}
	t.Fatalf("bundle has no %q entry", name)
	return nil
}
