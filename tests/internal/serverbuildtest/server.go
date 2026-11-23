// Package serverbuildtest owns package-lifetime Server binaries for process tests.
package serverbuildtest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// Cache builds lazily and retains both success and failure for one package run.
// It must not be copied after first use. Close runs after all tests have joined.
type Cache struct {
	// DisableCGO preserves the static build required by scratch-container tests.
	DisableCGO bool
	// TargetOS and TargetArch select the container engine target; empty uses Go defaults.
	TargetOS, TargetArch string
	// Package defaults to the Server; a separate cache can build an owned fixture probe.
	Package   string
	once      sync.Once
	directory string
	binary    string
	err       error
	build     func(root, output string, disableCGO bool) error
}

// Binary returns the same executable to every caller, including concurrent ones.
// The first repository root owns this cache; use one cache per package/build mode.
func (c *Cache) Binary(root string) (string, error) {
	c.once.Do(func() {
		c.directory, c.err = os.MkdirTemp("", "contractor-e2e-server-")
		if c.err != nil {
			return
		}
		c.binary = filepath.Join(c.directory, "contractor-server")
		if c.build != nil {
			c.err = c.build(root, c.binary, c.DisableCGO)
		} else {
			c.err = buildServer(root, c.binary, c.DisableCGO, c.TargetOS, c.TargetArch, c.Package)
		}
	})
	if c.err != nil {
		return "", c.err
	}
	return c.binary, nil
}

// Close removes the binary after the package has finished using it.
func (c *Cache) Close() error {
	if c.directory == "" {
		return nil
	}
	return os.RemoveAll(c.directory)
}

func buildServer(root, output string, disableCGO bool, targetOS, targetArch, packagePath string) error {
	start := time.Now()
	if packagePath == "" {
		packagePath = "./cmd/contractor-server"
	}
	cmd := exec.Command("go", "build", "-o", output, packagePath)
	cmd.Dir = root
	if disableCGO || targetOS != "" || targetArch != "" {
		cmd.Env = os.Environ()
		if disableCGO {
			cmd.Env = append(cmd.Env, "CGO_ENABLED=0")
		}
		if targetOS != "" {
			cmd.Env = append(cmd.Env, "GOOS="+targetOS)
		}
		if targetArch != "" {
			cmd.Env = append(cmd.Env, "GOARCH="+targetArch)
		}
	}
	data, err := cmd.CombinedOutput()
	fmt.Fprintf(os.Stderr, "e2e Server build: disableCGO=%t elapsed=%s\n", disableCGO, time.Since(start))
	if err != nil {
		return fmt.Errorf("build e2e Server: %w\n%s", err, data)
	}
	return nil
}
