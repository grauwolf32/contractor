//go:build integration

package gitimport

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

// The pipeline supplies a static Linux probe; the import itself executes
// inside a read-only container with no writable /tmp, home, checkout or PVC.
func TestRealGitReadOnlyContainer(t *testing.T) {
	probePath := os.Getenv("CONTRACTOR_TEST_GIT_PROBE_BINARY")
	if probePath == "" {
		t.Skip("Linux container proof runs through make test-git-read-only")
	}
	if _, err := exec.LookPath("podman"); err != nil {
		t.Fatal("read-only proof requires Podman")
	}
	repo, commit := realRepository(t)
	server := serveGitHTTP(t, repo, nil)
	defer server.Close()
	dir := t.TempDir()
	if err := os.Chmod(dir, 0755); err != nil {
		t.Fatal(err)
	}
	cert := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
	// Generate the fixture key independently so the probe can parse its in-memory
	// representation from a read-only fixture mount.
	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	owner, err := ssh.NewSignerFromKey(key)
	if err != nil {
		t.Fatal(err)
	}
	address, host := serveGitSSH(t, repo, owner)
	keyBlock, err := ssh.MarshalPrivateKey(key, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{"ca.pem": cert, "key.pem": pem.EncodeToMemory(keyBlock), "known_hosts": []byte(knownhosts.Line([]string{address}, host.PublicKey()) + "\n")} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
			t.Fatal(err)
		}
	}
	probe, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatalf("read pipeline probe: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "probe"), probe, 0755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "podman", "create", "--pull=never", "--network=host", "--read-only", "--read-only-tmpfs=false", "--user=65534:65534", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--env=GIT_PROBE_HTTPS="+server.URL+"/repo.git", "--env=GIT_PROBE_SSH=ssh://git@"+address+"/repo.git", "--env=GIT_PROBE_COMMIT="+commit, "--entrypoint=/fixture/probe", "docker.io/library/alpine:latest", "-test.run=^TestReadOnlyGitProbe$", "-test.v")
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("create read-only probe: %v: %s", err, output)
	}
	containerID := strings.TrimSpace(string(output))
	t.Cleanup(func() {
		cleanupCtx, cancelCleanup := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancelCleanup()
		if output, err := exec.CommandContext(cleanupCtx, "podman", "rm", "--force", "--time=0", containerID).CombinedOutput(); err != nil {
			t.Errorf("remove probe container: %v: %s", err, output)
		}
	})
	// Stage the fixture before startup; its files become part of the read-only root.
	if output, err := exec.CommandContext(ctx, "podman", "cp", dir, containerID+":/fixture").CombinedOutput(); err != nil {
		t.Fatalf("copy probe fixture: %v: %s", err, output)
	}
	output, err = exec.CommandContext(ctx, "podman", "start", "--attach", containerID).CombinedOutput()
	if err != nil {
		t.Fatalf("read-only probe: %v: %s", err, output)
	}
	t.Logf("%s", output)
}

func TestReadOnlyGitProbe(t *testing.T) {
	url := os.Getenv("GIT_PROBE_HTTPS")
	if url == "" {
		t.Skip("container subprocess only")
	}
	for _, path := range []string{"/tmp/git-probe", "/git-probe"} {
		file, err := os.Create(path)
		if err == nil {
			file.Close()
			t.Fatalf("unexpected writable path %s", path)
		}
		if !errors.Is(err, syscall.EROFS) && !errors.Is(err, syscall.EACCES) {
			t.Fatalf("write probe: %v", err)
		}
	}
	ca, err := os.ReadFile("/fixture/ca.pem")
	if err != nil {
		t.Fatal(err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(ca) {
		t.Fatal("fixture CA")
	}
	key, err := os.ReadFile("/fixture/key.pem")
	if err != nil {
		t.Fatal(err)
	}
	signer, err := ssh.ParsePrivateKey(key)
	clear(key)
	if err != nil {
		t.Fatal(err)
	}
	for _, url := range []string{url, os.Getenv("GIT_PROBE_SSH")} {
		remote, err := ParseRemote(url)
		if err != nil {
			t.Fatal(err)
		}
		client, _ := NewClient(Config{AllowedRemotes: []string{remote.Address}, KnownHostsFile: "/fixture/known_hosts"}, nil)
		client.allowLoopback = true
		client.tlsConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
		snapshot, err := client.Fetch(context.Background(), remote, "", ModeSnapshot, signer)
		if err != nil {
			t.Fatal(err)
		}
		assertSnapshot(t, snapshot, os.Getenv("GIT_PROBE_COMMIT"))
	}
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		t.Fatal(err)
	}
	t.Logf("HTTPS and private SSH succeeded; peak RSS=%d KiB", usage.Maxrss)
}
