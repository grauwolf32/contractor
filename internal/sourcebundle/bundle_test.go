package sourcebundle

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

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

func TestSourcePathsAgreeWithRuntimeFixture(t *testing.T) {
	encoded, err := os.ReadFile(filepath.Join("..", "..", "testdata", "source_member_paths.json"))
	if err != nil {
		t.Fatal(err)
	}
	var paths struct {
		Valid   []string `json:"valid"`
		Invalid []string `json:"invalid"`
	}
	if err := json.Unmarshal(encoded, &paths); err != nil {
		t.Fatal(err)
	}
	for _, path := range paths.Valid {
		if _, err := portablePath(path); err != nil {
			t.Errorf("valid runtime path %q: %v", path, err)
		}
	}
	for _, path := range paths.Invalid {
		if _, err := portablePath(path); err == nil {
			t.Errorf("runtime rejects path %q but sourcebundle accepts it", path)
		}
	}
}

func TestBuildRejectsColonInAnyPathComponent(t *testing.T) {
	for _, path := range []string{"logs/2024-01-01T10:00.txt", "a:b/c.txt"} {
		t.Run(path, func(t *testing.T) {
			root := t.TempDir()
			writeTestFile(t, filepath.Join(root, filepath.FromSlash(path)), "source")
			_, err := Build(root, Options{IncludeIgnored: true})
			if err == nil || !strings.Contains(err.Error(), path) || !strings.Contains(err.Error(), ".contractorignore") {
				t.Fatalf("Build error = %v, want path and exclusion remedy", err)
			}
		})
	}
}

func TestBuildRejectsEmptyAfterIgnoreFiltering(t *testing.T) {
	for _, ignored := range []bool{false, true} {
		t.Run(map[bool]string{false: "empty", true: "fully ignored"}[ignored], func(t *testing.T) {
			root := t.TempDir()
			if ignored {
				writeTestFile(t, filepath.Join(root, ".contractorignore"), "*\n")
				writeTestFile(t, filepath.Join(root, "main.go"), "package main")
			}
			if _, err := Build(root, Options{IncludeIgnored: true}); err == nil ||
				!strings.Contains(err.Error(), "no files to package") {
				t.Fatalf("Build error = %v, want empty source rejection", err)
			}
		})
	}
}

func TestBuildRejectsSymlink(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "target"), "target")
	if err := os.Symlink("target", filepath.Join(root, "link")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	_, err := Build(root, Options{IncludeIgnored: true})
	if err == nil || !strings.Contains(err.Error(), "link is a symbolic link") || !strings.Contains(err.Error(), ".contractorignore") {
		t.Fatalf("symlink error = %v, want the path and the .contractorignore remedy", err)
	}
	writeTestFile(t, filepath.Join(root, ".contractorignore"), "/link\n")
	bundle, err := Build(root, Options{IncludeIgnored: true})
	if err != nil {
		t.Fatal(err)
	}
	if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, []string{".contractorignore", "target"}) {
		t.Fatalf("paths = %v", paths)
	}
}

func TestBuildRejectsTrackedFilesThroughIgnoredSymlinkedParent(t *testing.T) {
	for _, location := range []string{"outside", "inside"} {
		t.Run(location, func(t *testing.T) {
			root := initGitRepository(t)
			tracked := filepath.Join(root, "vendor", "lib.txt")
			writeTestFile(t, tracked, "tracked content")
			gitAdd(t, root, "vendor/lib.txt")
			if err := os.Remove(tracked); err != nil {
				t.Fatal(err)
			}
			if err := os.Remove(filepath.Dir(tracked)); err != nil {
				t.Fatal(err)
			}
			targetDir := t.TempDir()
			if location == "inside" {
				targetDir = filepath.Join(root, "actual")
			}
			writeTestFile(t, filepath.Join(targetDir, "lib.txt"), "OUTSIDE SECRET")
			if err := os.Symlink(targetDir, filepath.Join(root, "vendor")); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
			writeTestFile(t, filepath.Join(root, ".git", "info", "exclude"), "/vendor\n")

			_, err := Build(root, Options{})
			if err == nil || !strings.Contains(err.Error(), "vendor is a symbolic link") || !strings.Contains(err.Error(), ".contractorignore") {
				t.Fatalf("Build error = %v, want rejection naming the symlinked parent", err)
			}
			if location == "outside" {
				writeTestFile(t, filepath.Join(root, ".contractorignore"), "/vendor/\n")
				bundle, err := Build(root, Options{})
				if err != nil {
					t.Fatalf("Build with .contractorignore remedy: %v", err)
				}
				if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, []string{".contractorignore"}) {
					t.Fatalf("bundle paths = %v, want only .contractorignore", paths)
				}
			}
		})
	}
}

func TestCopyRegularFileCannotEscapeAfterParentChanges(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, filepath.Join(root, "vendor", "lib.txt"), "inside")
	sourceRoot, err := os.OpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer sourceRoot.Close()
	files, _, err := inspectFiles(sourceRoot, []string{"vendor/lib.txt"})
	if err != nil || len(files) != 1 {
		t.Fatalf("inspect files = %v, %v", files, err)
	}
	if err := os.Rename(filepath.Join(root, "vendor"), filepath.Join(root, "old-vendor")); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	writeTestFile(t, filepath.Join(outside, "lib.txt"), "OUTSIDE SECRET")
	if err := os.Symlink(outside, filepath.Join(root, "vendor")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	var copied bytes.Buffer
	if err := copyRegularFile(&copied, sourceRoot, files[0]); err == nil || copied.Len() != 0 {
		t.Fatalf("copy after parent swap = %q, %v; want no outside bytes", copied.String(), err)
	}
}

func TestEncodeFilesRejectsFilesThatChangeSizeAfterInspection(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(t *testing.T, path string)
	}{
		{"appended past the per-file limit", func(t *testing.T, path string) {
			file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			if _, err := file.Write(bytes.Repeat([]byte("x"), 5<<20)); err != nil {
				t.Fatal(err)
			}
		}},
		{"truncated", func(t *testing.T, path string) {
			if err := os.Truncate(path, 2); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "app.log")
			writeTestFile(t, path, "start")
			sourceRoot, err := os.OpenRoot(root)
			if err != nil {
				t.Fatal(err)
			}
			defer sourceRoot.Close()
			files, _, err := inspectFiles(sourceRoot, []string{"app.log"})
			if err != nil || len(files) != 1 {
				t.Fatalf("inspect files = %v, %v", files, err)
			}
			test.change(t, path)
			if _, err := encodeFiles(sourceRoot, files); err == nil || !strings.Contains(err.Error(), "app.log changed while packaging") {
				t.Fatalf("encode after size change = %v, want the member to fail", err)
			}
		})
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

func TestBuildAppliesContractorIgnoreDespiteGitignoreMatches(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, ".gitignore"), "*.env\n!public.key\nbuild/\n")
	writeTestFile(t, filepath.Join(root, ".contractorignore"), "secret.env\npublic.key\ncache/\n")
	writeTestFile(t, filepath.Join(root, "nested", ".gitignore"), "!*.env\n")
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	writeTestFile(t, filepath.Join(root, "secret.env"), "overlapping .gitignore match")
	writeTestFile(t, filepath.Join(root, "public.key"), "overridden by a .gitignore negation")
	writeTestFile(t, filepath.Join(root, "nested", "secret.env"), "re-included by a nested .gitignore")
	writeTestFile(t, filepath.Join(root, "nested", "cache", "entry"), "nested directory pattern")
	writeTestFile(t, filepath.Join(root, "build", "cache", "output"), "inside a Git-ignored directory")
	writeTestFile(t, filepath.Join(root, "build", "kept.txt"), "force-added")
	gitAdd(t, root, "-f", "secret.env", "public.key", "nested/secret.env", "build/cache/output", "build/kept.txt")
	expected := []string{".contractorignore", ".gitignore", "build/kept.txt", "kept.go", "nested/.gitignore"}
	assertBundlePaths(t, root, expected)

	bundle, err := Build(root, Options{IncludeIgnored: true})
	if err != nil {
		t.Fatal(err)
	}
	if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, expected) {
		t.Fatalf("--include-ignored paths = %v, want %v", paths, expected)
	}
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

func TestBuildWalkSkipsNestedGitMetadataAndRepositories(t *testing.T) {
	root := t.TempDir()
	if insideGitWorkTree(root) {
		t.Skip("temporary directory is inside a Git work tree")
	}
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	writeTestFile(t, filepath.Join(root, "nested", ".git", "config"), "token in remote URL")
	writeTestFile(t, filepath.Join(root, "nested", "main.go"), "package nested")
	writeTestFile(t, filepath.Join(root, "mixed", ".GiT", "config"), "mixed-case metadata")
	writeTestFile(t, filepath.Join(root, "mixed", "main.go"), "package mixed")
	writeTestFile(t, filepath.Join(root, "linked", ".git"), "gitdir: ../outside")
	writeTestFile(t, filepath.Join(root, "linked", "main.go"), "package linked")
	for _, options := range []Options{{}, {IncludeIgnored: true}} {
		bundle, err := Build(root, options)
		if err != nil {
			t.Fatal(err)
		}
		if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, []string{"kept.go"}) {
			t.Fatalf("options %#v: paths = %v", options, paths)
		}
		if bundle.SkippedRepositories != 3 {
			t.Fatalf("options %#v: skipped repositories = %d, want 3", options, bundle.SkippedRepositories)
		}
	}
}

func TestBuildIncludeIgnoredSkipsNestedGitRepository(t *testing.T) {
	root := initGitRepository(t)
	writeTestFile(t, filepath.Join(root, "kept.go"), "package kept")
	nested := filepath.Join(root, "nested")
	if output, err := exec.Command("git", "init", "--quiet", nested).CombinedOutput(); err != nil {
		t.Fatalf("git init nested: %v: %s", err, output)
	}
	writeTestFile(t, filepath.Join(nested, "main.go"), "package nested")
	for _, options := range []Options{{}, {IncludeIgnored: true}} {
		bundle, err := Build(root, options)
		if err != nil {
			t.Fatal(err)
		}
		if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, []string{"kept.go"}) {
			t.Fatalf("options %#v: paths = %v", options, paths)
		}
		if bundle.SkippedRepositories != 1 {
			t.Fatalf("options %#v: skipped repositories = %d, want 1", options, bundle.SkippedRepositories)
		}
	}
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
	if paths := bundleEntryNames(t, bundle); !reflect.DeepEqual(paths, expected) {
		t.Fatalf("paths = %v, want %v", paths, expected)
	}
	return bundle
}

func bundleEntryNames(t *testing.T, bundle Bundle) []string {
	t.Helper()
	reader, err := zip.NewReader(bytes.NewReader(bundle.Data), int64(len(bundle.Data)))
	if err != nil {
		t.Fatal(err)
	}
	paths := make([]string, 0, len(reader.File))
	for _, file := range reader.File {
		paths = append(paths, file.Name)
	}
	sort.Strings(paths)
	return paths
}

func gitAdd(t *testing.T, root string, arguments ...string) {
	t.Helper()
	command := exec.Command("git", append([]string{"-C", root, "add"}, arguments...)...)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, output)
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
