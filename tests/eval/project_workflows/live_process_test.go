package projectworkflows

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const liveRetainedLogBytes = 64 << 10

type liveBoundedLog struct {
	mu   sync.Mutex
	data []byte
}

func (b *liveBoundedLog) Write(value []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	written := len(value)
	b.data = append(b.data, value...)
	if len(b.data) > liveRetainedLogBytes {
		b.data = append([]byte(nil), b.data[len(b.data)-liveRetainedLogBytes:]...)
	}
	return written, nil
}

type liveProcess struct {
	name string
	cmd  *exec.Cmd
	logs *liveBoundedLog
	done chan struct{}
	err  error
}

func startLiveProcess(
	t *testing.T,
	name, directory string,
	environment map[string]string,
	command string,
	args ...string,
) *liveProcess {
	t.Helper()
	logs := &liveBoundedLog{}
	cmd := exec.Command(command, args...)
	cmd.Dir = directory
	cmd.Env = liveEnvironment(environment)
	cmd.Stdout = logs
	cmd.Stderr = logs
	if err := cmd.Start(); err != nil {
		t.Fatalf("start %s: %s", name, safeErrorType(err))
	}
	process := &liveProcess{name: name, cmd: cmd, logs: logs, done: make(chan struct{})}
	go func() {
		process.err = cmd.Wait()
		close(process.done)
	}()
	t.Cleanup(func() { process.stop(t) })
	return process
}

func (p *liveProcess) stop(t *testing.T) {
	t.Helper()
	select {
	case <-p.done:
		return
	default:
	}
	if err := p.cmd.Process.Signal(syscall.SIGTERM); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Logf("signal %s failed (%s)", p.name, safeErrorType(err))
	}
	select {
	case <-p.done:
		return
	case <-time.After(15 * time.Second):
	}
	if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		t.Logf("kill %s failed (%s)", p.name, safeErrorType(err))
	}
	select {
	case <-p.done:
	case <-time.After(5 * time.Second):
		t.Logf("%s did not report process exit", p.name)
	}
}

func (p *liveProcess) exited() (bool, error) {
	select {
	case <-p.done:
		return true, p.err
	default:
		return false, nil
	}
}

func runLiveChecked(
	t *testing.T,
	directory string,
	environment map[string]string,
	command string,
	args ...string,
) {
	t.Helper()
	cmd := exec.Command(command, args...)
	cmd.Dir = directory
	cmd.Env = liveEnvironment(environment)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		t.Fatalf("run %s failed (%s)", filepath.Base(command), safeErrorType(err))
	}
}

func liveEnvironment(overrides map[string]string) []string {
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

func waitForLiveHTTP(
	t *testing.T,
	ctx context.Context,
	process *liveProcess,
	client *http.Client,
	target string,
) {
	t.Helper()
	for {
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
		response, err := client.Do(request)
		if err == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return
			}
		}
		if exited, processErr := process.exited(); exited {
			t.Fatalf("%s exited during readiness (%s)", process.name, safeErrorType(processErr))
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for %s readiness exceeded its deadline", process.name)
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func newLiveMTLSClient(t *testing.T, caFile string, identity localpki.Paths) *http.Client {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(identity.Certificate, identity.PrivateKey)
	if err != nil {
		t.Fatal("load live evaluation mTLS identity")
	}
	return &http.Client{
		Transport: &http.Transport{TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS13,
			RootCAs:    liveCertificatePool(t, caFile),
			Certificates: []tls.Certificate{
				certificate,
			},
		}},
		Timeout: 10 * time.Second,
	}
}

func liveCertificatePool(t *testing.T, caFile string) *x509.CertPool {
	t.Helper()
	data, err := os.ReadFile(caFile)
	if err != nil {
		t.Fatal("read live evaluation CA")
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(data) {
		t.Fatal("live evaluation CA contains no certificate")
	}
	return pool
}

func liveFreeAddress(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal("allocate loopback address")
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal("release loopback address")
	}
	return address
}

func liveIsolatedDatabase(t *testing.T, ctx context.Context, databaseURL string) string {
	t.Helper()
	parsed, err := url.Parse(databaseURL)
	if err != nil || parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL must be a PostgreSQL URL")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal("open live evaluation PostgreSQL")
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal("ping live evaluation PostgreSQL")
	}
	schema := "contractor_live_eval_" + liveRandomHex(t, 8)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+identifier); err != nil {
		admin.Close()
		t.Fatal("create live evaluation schema")
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_, _ = admin.Exec(cleanup, "DROP SCHEMA "+identifier+" CASCADE")
		admin.Close()
	})
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func liveRandomHex(t *testing.T, size int) string {
	t.Helper()
	data := make([]byte, size)
	if _, err := rand.Read(data); err != nil {
		t.Fatal("generate live evaluation identifier")
	}
	return hex.EncodeToString(data)
}

func liveRepositoryRoot(t *testing.T) string {
	t.Helper()
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate live evaluation source")
	}
	root, err := filepath.Abs(filepath.Join(filepath.Dir(source), "..", "..", ".."))
	if err != nil {
		t.Fatal("resolve repository root")
	}
	return root
}

func copyLiveConfiguration(t *testing.T, repositoryRoot, target, model, gatewayURL string) {
	t.Helper()
	source := filepath.Join(repositoryRoot, "configs")
	gatewayUpdated := false
	workflowUpdates := 0
	err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		destination := filepath.Join(target, relative)
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("configuration contains a symbolic link")
		}
		if entry.IsDir() {
			return os.MkdirAll(destination, 0o700)
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("configuration contains a non-regular entry")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if relative == filepath.Join("model-policies", "domain_worker.yaml") {
			encoded, _ := jsonString(model)
			updated := strings.Replace(string(data), "model: worker-model", "model: "+encoded, 1)
			if updated == string(data) {
				return fmt.Errorf("domain Worker model marker is absent")
			}
			data = []byte(updated)
		}
		if relative == filepath.Join("llm-gateways", "local_litellm.yaml") {
			encoded, _ := jsonString(gatewayURL)
			lines := strings.Split(string(data), "\n")
			for index, line := range lines {
				if strings.HasPrefix(line, "  url: ") {
					lines[index] = "  url: " + encoded
					gatewayUpdated = true
				}
			}
			data = []byte(strings.Join(lines, "\n"))
		}
		if strings.HasPrefix(relative, "workflows"+string(filepath.Separator)) {
			const gatewaySelection = "      llmGateway: local-litellm@1\n"
			const credentialSelection = gatewaySelection + "      credential: development-worker\n"
			updated := strings.ReplaceAll(string(data), gatewaySelection, credentialSelection)
			if updated != string(data) {
				workflowUpdates++
				data = []byte(updated)
			}
		}
		return os.WriteFile(destination, data, 0o600)
	})
	if err != nil {
		t.Fatalf("copy live evaluation configuration (%s)", safeErrorType(err))
	}
	if !gatewayUpdated || workflowUpdates == 0 {
		t.Fatalf("live config updates = gateway:%t workflows:%d", gatewayUpdated, workflowUpdates)
	}
}

func jsonString(value string) (string, error) {
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return "", err
	}
	return strings.TrimSpace(output.String()), nil
}

func liveWorkRootEmpty(root string) bool {
	entries, err := os.ReadDir(root)
	return errors.Is(err, os.ErrNotExist) || err == nil && len(entries) == 0
}

func safeErrorType(err error) string {
	if err == nil {
		return "none"
	}
	return fmt.Sprintf("%T", err)
}
