package serverbuildtest

import (
	"debug/elf"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func TestConcurrentCallersShareBuildAndPackageLifetime(t *testing.T) {
	var builds atomic.Int32
	cache := Cache{build: func(root, output string, disableCGO bool) error {
		builds.Add(1)
		if root != "repository" || disableCGO {
			t.Error("build configuration changed")
		}
		return os.WriteFile(output, []byte("executable"), 0700)
	}}
	t.Cleanup(func() { _ = cache.Close() })
	var callers sync.WaitGroup
	paths := make(chan string, 32)
	for range 32 {
		callers.Go(func() {
			path, err := cache.Binary("repository")
			if err != nil {
				t.Error(err)
				return
			}
			paths <- path
		})
	}
	callers.Wait()
	close(paths)
	var binary string
	for path := range paths {
		if binary != "" && path != binary {
			t.Error("callers received different binaries")
		}
		binary = path
	}
	if builds.Load() != 1 {
		t.Fatalf("builds = %d", builds.Load())
	}
	if _, err := os.Stat(binary); err != nil {
		t.Fatal(err)
	}
	// Finishing one scenario must not remove a package-owned executable.
	t.Run("first scenario finishes", func(t *testing.T) { t.TempDir() })
	next, err := cache.Binary("repository")
	if err != nil || next != binary {
		t.Fatalf("reuse: %q %v", next, err)
	}
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Dir(binary)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("package directory retained: %v", err)
	}
}

func TestFailedBuildIsNotRetriedAndPartialOutputIsRemoved(t *testing.T) {
	failure := errors.New("compiler failed")
	builds := 0
	cache := Cache{build: func(_, output string, _ bool) error {
		builds++
		if err := os.WriteFile(output, []byte("partial"), 0700); err != nil {
			t.Fatal(err)
		}
		return failure
	}}
	t.Cleanup(func() { _ = cache.Close() })
	for range 3 {
		path, err := cache.Binary("repository")
		if path != "" || !errors.Is(err, failure) {
			t.Fatalf("failure: %q %v", path, err)
		}
	}
	if builds != 1 {
		t.Fatalf("retried failed build %d times", builds)
	}
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(cache.directory); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("partial build retained: %v", err)
	}
}

func TestUnusedCacheDoesNotBuildOrAllocate(t *testing.T) {
	cache := Cache{build: func(_, _ string, _ bool) error { t.Fatal("unexpected build"); return nil }}
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
	if cache.directory != "" {
		t.Fatal("unused cache allocated a directory")
	}
}

func TestBuildModesStayDistinct(t *testing.T) {
	for _, disableCGO := range []bool{false, true} {
		cache := Cache{DisableCGO: disableCGO, build: func(_, output string, actual bool) error {
			if actual != disableCGO {
				t.Fatalf("disableCGO = %t", actual)
			}
			return os.WriteFile(output, nil, 0700)
		}}
		if _, err := cache.Binary("repository"); err != nil {
			t.Fatal(err)
		}
		if err := cache.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGoCommandPreservesBuildArgumentsAndCGOMode(t *testing.T) {
	toolDirectory := t.TempDir()
	compiler := `#!/bin/sh
if [ "$1" != build ] || [ "$2" != -o ] || [ "$4" != ./cmd/contractor-server ] || [ "$#" != 4 ]; then
 exit 3
fi
printf '%s' "$CGO_ENABLED" > "$3"
`
	if err := os.WriteFile(filepath.Join(toolDirectory, "go"), []byte(compiler), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", toolDirectory)
	t.Setenv("CGO_ENABLED", "1")
	for _, disabled := range []bool{false, true} {
		cache := Cache{DisableCGO: disabled}
		t.Cleanup(func() { _ = cache.Close() })
		binary, err := cache.Binary(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(binary)
		if err != nil {
			t.Fatal(err)
		}
		want := "1"
		if disabled {
			want = "0"
		}
		if string(data) != want {
			t.Fatalf("CGO_ENABLED = %q, want %q", data, want)
		}
	}
}

// Exercise the real compiler: a native process executable and a Linux image
// artifact are distinct even when the selected engine is on another platform.
func TestRealNativeAndLinuxTargets(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "cmd", "contractor-server"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "cmd", "contractor-server", "main.go"), []byte("package main\nfunc main(){println(\"native\")}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	var native Cache
	defer native.Close()
	executable, err := native.Binary(root)
	if err != nil {
		t.Fatal(err)
	}
	output, err := exec.Command(executable).CombinedOutput()
	if err != nil || string(output) != "native\n" {
		t.Fatalf("native executable did not run: %v", err)
	}
	arch, machine := "amd64", elf.EM_X86_64
	if runtime.GOARCH == "amd64" {
		arch, machine = "arm64", elf.EM_AARCH64
	}
	linux := Cache{DisableCGO: true, TargetOS: "linux", TargetArch: arch}
	defer linux.Close()
	imageArtifact, err := linux.Binary(root)
	if err != nil {
		t.Fatal(err)
	}
	binary, err := elf.Open(imageArtifact)
	if err != nil {
		t.Fatal(err)
	}
	defer binary.Close()
	if binary.Machine != machine {
		t.Fatalf("unexpected ELF machine: %v", binary.Machine)
	}
}
