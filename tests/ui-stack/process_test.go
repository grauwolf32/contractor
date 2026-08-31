//go:build e2e

package uistack

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

const retainedProcessLogBytes = 128 << 10

type boundedLog struct {
	mu   sync.Mutex
	data []byte
}

func (b *boundedLog) Write(value []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	written := len(value)
	b.data = append(b.data, value...)
	if len(b.data) > retainedProcessLogBytes {
		b.data = append([]byte(nil), b.data[len(b.data)-retainedProcessLogBytes:]...)
	}
	return written, nil
}

func (b *boundedLog) contents() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(append([]byte(nil), b.data...))
}

func (b *boundedLog) redacted(secrets ...string) string {
	result := b.contents()
	for _, secret := range secrets {
		if secret != "" {
			result = strings.ReplaceAll(result, secret, "[REDACTED]")
		}
	}
	return result
}

type childProcess struct {
	name string
	cmd  *exec.Cmd
	logs *boundedLog
	done chan struct{}
	err  error
}

func startProcess(
	t *testing.T,
	name string,
	directory string,
	environment []string,
	command string,
	args ...string,
) *childProcess {
	t.Helper()
	process, err := launchProcess(name, directory, environment, command, args...)
	if err != nil {
		t.Fatalf("start %s: %v", name, err)
	}
	return process
}

func launchProcess(
	name string,
	directory string,
	environment []string,
	command string,
	args ...string,
) (*childProcess, error) {
	logs := &boundedLog{}
	cmd := exec.Command(command, args...)
	cmd.Dir = directory
	cmd.Env = environment
	cmd.Stdout = logs
	cmd.Stderr = logs
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	process := &childProcess{name: name, cmd: cmd, logs: logs, done: make(chan struct{})}
	go func() {
		process.err = cmd.Wait()
		close(process.done)
	}()
	return process, nil
}

func (p *childProcess) stop(t *testing.T) {
	t.Helper()
	if p == nil {
		return
	}
	select {
	case <-p.done:
		return
	default:
	}
	if err := p.cmd.Process.Signal(syscall.SIGTERM); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Logf("signal %s: %v", p.name, err)
	}
	select {
	case <-p.done:
		return
	case <-time.After(10 * time.Second):
	}
	if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Logf("kill %s: %v", p.name, err)
	}
	select {
	case <-p.done:
	case <-time.After(5 * time.Second):
		t.Logf("%s did not report process exit", p.name)
	}
}

func (p *childProcess) exited() (bool, error) {
	select {
	case <-p.done:
		return true, p.err
	default:
		return false, nil
	}
}

func runChecked(
	t *testing.T,
	directory string,
	environment []string,
	command string,
	args ...string,
) string {
	t.Helper()
	var output bytes.Buffer
	cmd := exec.Command(command, args...)
	cmd.Dir = directory
	cmd.Env = environment
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Run(); err != nil {
		t.Fatalf("run %s: %v\n%s", command, err, output.String())
	}
	return output.String()
}

func inheritedEnvironment(overrides map[string]string) []string {
	blocked := make(map[string]struct{}, len(overrides))
	for key := range overrides {
		blocked[key] = struct{}{}
	}
	result := make([]string, 0, len(os.Environ())+len(overrides))
	for _, item := range os.Environ() {
		key, _, _ := strings.Cut(item, "=")
		if _, replace := blocked[key]; !replace {
			result = append(result, item)
		}
	}
	return appendOverrides(result, overrides)
}

func cleanEnvironment(overrides map[string]string, retained ...string) []string {
	result := make([]string, 0, len(retained)+len(overrides))
	for _, key := range retained {
		if value, exists := os.LookupEnv(key); exists {
			result = append(result, key+"="+value)
		}
	}
	return appendOverrides(result, overrides)
}

func appendOverrides(result []string, overrides map[string]string) []string {
	keys := make([]string, 0, len(overrides))
	for key := range overrides {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result = append(result, key+"="+overrides[key])
	}
	return result
}

func copyBounded(reader io.Reader) string {
	data, err := io.ReadAll(io.LimitReader(reader, retainedProcessLogBytes+1))
	if err != nil {
		return fmt.Sprintf("read response: %v", err)
	}
	if len(data) > retainedProcessLogBytes {
		data = data[:retainedProcessLogBytes]
	}
	return string(data)
}
