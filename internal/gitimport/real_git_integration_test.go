//go:build integration

package gitimport

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

func nativeGit(t *testing.T, repo string, args ...string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "git", append([]string{"-C", repo}, args...)...)
	command.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_AUTHOR_NAME=Fixture", "GIT_AUTHOR_EMAIL=fixture@example.test", "GIT_COMMITTER_NAME=Fixture", "GIT_COMMITTER_EMAIL=fixture@example.test")
	data, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("Git fixture %v: %v: %s", args, err, data)
	}
	return data
}
func realRepository(t *testing.T) (string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Fatal("integration fixture requires native git")
	}
	repo := t.TempDir()
	nativeGit(t, repo, "init", "--initial-branch=main")
	if err := os.WriteFile(filepath.Join(repo, "hello.txt"), []byte("original\n"), 0600); err != nil {
		t.Fatal(err)
	}
	// Similar blobs exercise Git's normal OFS delta selection in upload-pack.
	for i := 0; i < 12; i++ {
		if err := os.WriteFile(filepath.Join(repo, fmt.Sprintf("file-%02d.txt", i)), []byte(strings.Repeat("shared content\n", 10000)+fmt.Sprint(i)), 0600); err != nil {
			t.Fatal(err)
		}
	}
	nativeGit(t, repo, "add", ".")
	nativeGit(t, repo, "commit", "-m", "fixture")
	nativeGit(t, repo, "tag", "-a", "v1", "-m", "release")
	return repo, strings.TrimSpace(string(nativeGit(t, repo, "rev-parse", "HEAD")))
}
func serveGitHTTP(t *testing.T, repo string, beforeFetch func()) *httptest.Server {
	t.Helper()
	return httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		args := []string{"upload-pack", "--stateless-rpc"}
		switch r.URL.Path {
		case "/repo.git/info/refs":
			if r.Method != "GET" || r.URL.RawQuery != "service=git-upload-pack" {
				w.WriteHeader(400)
				return
			}
			w.Header().Set("Content-Type", "application/x-git-upload-pack-advertisement")
			_, _ = io.WriteString(w, "001e# service=git-upload-pack\n0000")
			args = append(args, "--advertise-refs")
		case "/repo.git/git-upload-pack":
			if r.Method != "POST" {
				w.WriteHeader(405)
				return
			}
			w.Header().Set("Content-Type", "application/x-git-upload-pack-result")
			if beforeFetch != nil {
				beforeFetch()
			}
		default:
			w.WriteHeader(404)
			return
		}
		command := exec.CommandContext(r.Context(), "git", append(args, repo)...)
		command.Stdin = r.Body
		command.Stdout = w
		if err := command.Run(); err != nil {
			t.Errorf("upload-pack: %v", err)
		}
	}))
}
func assertSnapshot(t *testing.T, snapshot Snapshot, commit string) {
	t.Helper()
	if snapshot.Commit != commit {
		t.Fatalf("commit %s != %s", snapshot.Commit, commit)
	}
	reader, err := zip.NewReader(bytes.NewReader(snapshot.Data), int64(len(snapshot.Data)))
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, file := range reader.File {
		if file.Name == "hello.txt" {
			r, _ := file.Open()
			data, err := io.ReadAll(r)
			_ = r.Close()
			if err != nil || string(data) != "original\n" {
				t.Fatalf("snapshot moved: %q %v", data, err)
			}
			found = true
		}
	}
	if !found || len(reader.File) != 13 {
		t.Fatalf("incomplete snapshot: %d files", len(reader.File))
	}
}
func TestRealHTTPSPinnedCommitAndTags(t *testing.T) {
	repo, commit := realRepository(t)
	var once sync.Once
	server := serveGitHTTP(t, repo, func() {
		once.Do(func() {
			if err := os.WriteFile(filepath.Join(repo, "hello.txt"), []byte("branch advanced\n"), 0600); err != nil {
				t.Error(err)
				return
			}
			nativeGit(t, repo, "add", ".")
			nativeGit(t, repo, "commit", "-m", "advance")
		})
	})
	defer server.Close()
	remote, _ := ParseRemote(server.URL + "/repo.git")
	client, _ := NewClient(Config{AllowedRemotes: []string{remote.Address}})
	client.allowLoopback = true
	client.tlsConfig = server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	snapshot, err := client.Fetch(context.Background(), remote, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSnapshot(t, snapshot, commit)
	runtime.ReadMemStats(&after)
	t.Logf("real HTTPS fixture: ZIP=%d bytes, allocated=%d bytes, heap=%d bytes", len(snapshot.Data), after.TotalAlloc-before.TotalAlloc, after.HeapAlloc)
	for _, ref := range []string{"v1", "refs/tags/v1"} {
		tag, err := client.Fetch(context.Background(), remote, ref, nil)
		if err != nil {
			t.Fatalf("tag %s: %v", ref, err)
		}
		assertSnapshot(t, tag, commit)
		if !bytes.Equal(snapshot.Data, tag.Data) {
			t.Fatal("tag ZIP differs")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.Fetch(ctx, remote, "", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
}

func TestDecodedPackCumulativeBudget(t *testing.T) {
	// Each object is individually allowed; the fifth must be rejected before
	// inflation even though the compressed pack is tiny and objects deduplicate.
	entry := packEntry{kind: plumbing.BlobObject, data: make([]byte, maxObjectBytes)}
	pack := makePack(entry, entry, entry, entry, entry)
	if _, err := decodePack(context.Background(), pack); !errors.Is(err, ErrBudget) {
		t.Fatalf("cumulative decoded bytes: %v", err)
	}
}

func newSigner(t *testing.T) ssh.Signer {
	t.Helper()
	_, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	signer, err := ssh.NewSignerFromKey(private)
	if err != nil {
		t.Fatal(err)
	}
	return signer
}
func serveGitSSH(t *testing.T, repo string, owner ssh.Signer) (string, ssh.Signer) {
	t.Helper()
	host := newSigner(t)
	config := &ssh.ServerConfig{PublicKeyCallback: func(_ ssh.ConnMetadata, key ssh.PublicKey) (*ssh.Permissions, error) {
		if !bytes.Equal(key.Marshal(), owner.PublicKey().Marshal()) {
			return nil, errors.New("unauthorized")
		}
		return nil, nil
	}}
	config.AddHostKey(host)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
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
						if request.Type != "exec" || ssh.Unmarshal(request.Payload, &payload) != nil || (payload.Command != "git-upload-pack '/repo.git'" && payload.Command != "git-upload-pack 'relative/repo.git'") {
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
	return listener.Addr().String(), host
}
func TestRealSSHOwnerKeyAndStrictHostTrust(t *testing.T) {
	repo, commit := realRepository(t)
	owner := newSigner(t)
	address, host := serveGitSSH(t, repo, owner)
	known := filepath.Join(t.TempDir(), "known_hosts")
	if err := os.WriteFile(known, []byte(knownhosts.Line([]string{address}, host.PublicKey())+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	remote, _ := ParseRemote("ssh://git@" + address + "/repo.git")
	client, _ := NewClient(Config{AllowedRemotes: []string{address}, KnownHostsFile: known})
	client.allowLoopback = true
	snapshot, err := client.Fetch(context.Background(), remote, "main", owner)
	if err != nil {
		t.Fatal(err)
	}
	assertSnapshot(t, snapshot, commit)
	relative, err := ParseRemote("ssh://git@" + address + "/~/relative/repo.git")
	if err != nil {
		t.Fatal(err)
	}
	relativeSnapshot, err := client.Fetch(context.Background(), relative, "main", owner)
	if err != nil {
		t.Fatal(err)
	}
	assertSnapshot(t, relativeSnapshot, commit)
	if _, err := client.Fetch(context.Background(), remote, "main", newSigner(t)); !errors.Is(err, ErrRemote) {
		t.Fatalf("wrong owner: %v", err)
	}
	if err := os.WriteFile(known, []byte(knownhosts.Line([]string{address}, newSigner(t).PublicKey())+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Fetch(context.Background(), remote, "main", owner); !errors.Is(err, ErrTrust) {
		t.Fatalf("changed host: %v", err)
	}
}

// The parent creates fixtures and a static probe; the import itself executes
// inside a read-only container with no writable /tmp, home, checkout or PVC.
func TestRealGitReadOnlyContainer(t *testing.T) {
	if _, err := exec.LookPath("podman"); err != nil {
		t.Skip("read-only proof requires Podman")
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
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "test", "-c", "-tags=integration", "-o", filepath.Join(dir, "probe"), ".")
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build probe: %v: %s", err, output)
	}
	command := exec.CommandContext(ctx, "podman", "run", "--rm", "--pull=never", "--network=host", "--read-only", "--read-only-tmpfs=false", "--user=65534:65534", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--mount=type=bind,src="+dir+",dst=/fixture,ro=true", "--env=GIT_PROBE_HTTPS="+server.URL+"/repo.git", "--env=GIT_PROBE_SSH=ssh://git@"+address+"/repo.git", "--env=GIT_PROBE_COMMIT="+commit, "--entrypoint=/fixture/probe", "docker.io/library/alpine:latest", "-test.run=^TestReadOnlyGitProbe$", "-test.v")
	output, err := command.CombinedOutput()
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
		client, _ := NewClient(Config{AllowedRemotes: []string{remote.Address}, KnownHostsFile: "/fixture/known_hosts"})
		client.allowLoopback = true
		client.tlsConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
		snapshot, err := client.Fetch(context.Background(), remote, "", signer)
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
