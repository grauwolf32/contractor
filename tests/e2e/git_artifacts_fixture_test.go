//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"encoding/pem"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
)

func gitGateCommand(t *testing.T, repo string, args ...string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "git", append([]string{"-C", repo}, args...)...)
	command.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_AUTHOR_NAME=Fixture", "GIT_AUTHOR_EMAIL=fixture@example.test", "GIT_COMMITTER_NAME=Fixture", "GIT_COMMITTER_EMAIL=fixture@example.test")
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("Git fixture failed: %v: %s", err, output)
	}
	return strings.TrimSpace(string(output))
}
func gitGateRepository(t *testing.T) (string, string) {
	t.Helper()
	repo := t.TempDir()
	gitGateCommand(t, repo, "init", "--initial-branch=main")
	if err := os.WriteFile(filepath.Join(repo, "source.txt"), []byte("before\n"), 0600); err != nil {
		t.Fatal(err)
	}
	gitGateCommand(t, repo, "add", ".")
	gitGateCommand(t, repo, "commit", "-m", "initial")
	gitGateCommand(t, repo, "tag", "-a", "initial", "-m", "initial")
	return repo, gitGateCommand(t, repo, "rev-parse", "HEAD")
}
func gitGateHost(t *testing.T) string {
	t.Helper()
	addresses, err := net.InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	for _, address := range addresses {
		network, ok := address.(*net.IPNet)
		if ok && network.IP.To4() != nil && !network.IP.IsLoopback() && !network.IP.IsLinkLocalUnicast() {
			return network.IP.String()
		}
	}
	t.Fatal("Git container proof requires a non-loopback IPv4 interface")
	return ""
}

type gitGatePause struct {
	entered  chan struct{}
	released chan struct{}
}
type gitGateHTTPS struct {
	server *httptest.Server
	mu     sync.Mutex
	pause  *gitGatePause
}

func (g *gitGateHTTPS) blockNext() (chan struct{}, func()) {
	pause := &gitGatePause{entered: make(chan struct{}), released: make(chan struct{})}
	g.mu.Lock()
	g.pause = pause
	g.mu.Unlock()
	var once sync.Once
	return pause.entered, func() { once.Do(func() { close(pause.released) }) }
}
func gitGateServeHTTPS(t *testing.T, repo, host, certificate, key string) *gitGateHTTPS {
	t.Helper()
	gate := &gitGateHTTPS{}
	gate.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		args := []string{"upload-pack", "--stateless-rpc"}
		switch r.URL.Path {
		case "/repo.git/info/refs":
			w.Header().Set("Content-Type", "application/x-git-upload-pack-advertisement")
			_, _ = io.WriteString(w, "001e# service=git-upload-pack\n0000")
			args = append(args, "--advertise-refs")
		case "/repo.git/git-upload-pack":
			gate.mu.Lock()
			pause := gate.pause
			gate.pause = nil
			gate.mu.Unlock()
			if pause != nil {
				close(pause.entered)
				select {
				case <-pause.released:
				case <-r.Context().Done():
					return
				}
			}
			w.Header().Set("Content-Type", "application/x-git-upload-pack-result")
		default:
			w.WriteHeader(404)
			return
		}
		command := exec.CommandContext(r.Context(), "git", append(args, repo)...)
		command.Stdin = r.Body
		command.Stdout = w
		_ = command.Run() // A cancelled HTTP reader may close upload-pack early.
	}))
	_ = gate.server.Listener.Close()
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	gate.server.Listener = listener
	pair, err := tls.LoadX509KeyPair(certificate, key)
	if err != nil {
		t.Fatal(err)
	}
	gate.server.TLS = &tls.Config{Certificates: []tls.Certificate{pair}, MinVersion: tls.VersionTLS12}
	gate.server.StartTLS()
	t.Cleanup(gate.server.Close)
	return gate
}
func gitGateSigner(t *testing.T) (ssh.Signer, []byte) {
	t.Helper()
	_, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	signer, err := ssh.NewSignerFromKey(key)
	if err != nil {
		t.Fatal(err)
	}
	block, err := ssh.MarshalPrivateKey(key, "fixture-canary")
	if err != nil {
		t.Fatal(err)
	}
	return signer, pem.EncodeToMemory(block)
}
func gitGateServeSSH(t *testing.T, repo, host string, owner ssh.Signer) (string, ssh.Signer) {
	t.Helper()
	identity, _ := gitGateSigner(t)
	config := &ssh.ServerConfig{PublicKeyCallback: func(_ ssh.ConnMetadata, key ssh.PublicKey) (*ssh.Permissions, error) {
		if !bytes.Equal(key.Marshal(), owner.PublicKey().Marshal()) {
			return nil, errors.New("unauthorized")
		}
		return nil, nil
	}}
	config.AddHostKey(identity)
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	t.Cleanup(func() { cancel(); _ = listener.Close(); wg.Wait() })
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer conn.Close()
				stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
				defer stop()
				connection, channels, requests, err := ssh.NewServerConn(conn, config)
				if err != nil {
					return
				}
				defer connection.Close()
				go ssh.DiscardRequests(requests)
				for incoming := range channels {
					if incoming.ChannelType() != "session" {
						_ = incoming.Reject(ssh.UnknownChannelType, "unsupported")
						continue
					}
					channel, requests, err := incoming.Accept()
					if err != nil {
						return
					}
					for request := range requests {
						var payload struct{ Command string }
						if request.Type != "exec" || ssh.Unmarshal(request.Payload, &payload) != nil || payload.Command != "git-upload-pack '/repo.git'" {
							_ = request.Reply(false, nil)
							continue
						}
						_ = request.Reply(true, nil)
						command := exec.CommandContext(ctx, "git", "upload-pack", repo)
						stdin, err := command.StdinPipe()
						if err != nil {
							_ = channel.Close()
							return
						}
						go func() { _, _ = io.Copy(stdin, channel); _ = stdin.Close() }()
						command.Stdout = channel
						command.Stderr = channel.Stderr()
						code := uint32(0)
						if command.Run() != nil {
							code = 1
						}
						_, _ = channel.SendRequest("exit-status", false, ssh.Marshal(struct{ Status uint32 }{code}))
						_ = channel.Close()
						break
					}
				}
			}()
		}
	}()
	return listener.Addr().String(), identity
}
