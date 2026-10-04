package gitimport

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

func testSSHSigner(t *testing.T) ssh.Signer {
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

type repeatedByteReader struct{}

func (repeatedByteReader) Read(p []byte) (int, error) {
	for index := range p {
		p[index] = 'x'
	}
	return len(p), nil
}

func serveBudgetSSH(t *testing.T, owner ssh.Signer, payloadBytes int64) (Remote, string) {
	t.Helper()
	return serveBudgetSSHWithHost(t, owner, testSSHSigner(t), payloadBytes)
}

// serveBudgetSSHWithHost serves a fake upload-pack with only the given host
// key and pins that key in a fresh known_hosts file.
func serveBudgetSSHWithHost(t *testing.T, owner, host ssh.Signer, payloadBytes int64) (Remote, string) {
	t.Helper()
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
	var workers sync.WaitGroup
	workers.Add(1)
	go func() {
		defer workers.Done()
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			workers.Add(1)
			go func() {
				defer workers.Done()
				defer conn.Close()
				_, channels, requests, err := ssh.NewServerConn(conn, config)
				if err != nil {
					return
				}
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
						if request.Type != "exec" {
							_ = request.Reply(false, nil)
							continue
						}
						_ = request.Reply(true, nil)
						advertisement := strings.Repeat("1", 40) + " HEAD\x00shallow\n"
						_, _ = fmt.Fprintf(channel, "%04x%s0000", len(advertisement)+4, advertisement)
						var uploadRequest []byte
						chunk := make([]byte, 256)
						for !bytes.Contains(uploadRequest, []byte("0009done\n")) {
							n, err := channel.Read(chunk)
							if err != nil || len(uploadRequest)+n > 4096 {
								_ = channel.Close()
								return
							}
							uploadRequest = append(uploadRequest, chunk[:n]...)
						}
						_, _ = io.WriteString(channel, "00000008NAK\nPACK")
						if payloadBytes > 0 {
							_, _ = io.CopyN(channel, repeatedByteReader{}, payloadBytes)
						} else {
							_, _ = io.WriteString(channel, "bad")
						}
						_ = channel.Close()
						break
					}
				}
			}()
		}
	}()
	t.Cleanup(func() { _ = listener.Close(); workers.Wait() })
	address := listener.Addr().String()
	known := filepath.Join(t.TempDir(), "known_hosts")
	if err := os.WriteFile(known, []byte(knownhosts.Line([]string{address}, host.PublicKey())+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	remote, err := ParseRemote("ssh://git@" + address + "/repo.git")
	if err != nil {
		t.Fatal(err)
	}
	return remote, known
}

func TestSSHOversizedWireAlwaysReturnsBudget(t *testing.T) {
	owner := testSSHSigner(t)
	remote, known := serveBudgetSSH(t, owner, 1<<20)
	client, err := NewClient(Config{AllowedRemotes: []string{remote.Address}, KnownHostsFile: known}, nil)
	if err != nil {
		t.Fatal(err)
	}
	client.allowLoopback = true
	client.sshWireBudget = 128 << 10
	for iteration := range 20 {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err := client.Fetch(ctx, remote, "", owner)
		cancel()
		if !errors.Is(err, ErrBudget) {
			t.Fatalf("iteration %d: got %v, want ErrBudget", iteration, err)
		}
	}
	// Exercise the real production cap once as well as the repeated small-cap
	// transport race above. The server streams bytes without retaining a pack.
	remote, known = serveBudgetSSH(t, owner, MaxReceivedBytes+(8<<20))
	client, err = NewClient(Config{AllowedRemotes: []string{remote.Address}, KnownHostsFile: known}, nil)
	if err != nil {
		t.Fatal(err)
	}
	client.allowLoopback = true
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	_, err = client.Fetch(ctx, remote, "", owner)
	cancel()
	if !errors.Is(err, ErrBudget) {
		t.Fatalf("production wire limit: got %v, want ErrBudget", err)
	}
}

func TestSSHInvalidPackAndAuthFailureKeepTheirErrors(t *testing.T) {
	owner := testSSHSigner(t)
	remote, known := serveBudgetSSH(t, owner, 0)
	client, err := NewClient(Config{AllowedRemotes: []string{remote.Address}, KnownHostsFile: known}, nil)
	if err != nil {
		t.Fatal(err)
	}
	client.allowLoopback = true
	if _, err := client.Fetch(context.Background(), remote, "", owner); !errors.Is(err, ErrContent) {
		t.Fatalf("fitting invalid pack: %v", err)
	}
	if _, err := client.Fetch(context.Background(), remote, "", testSSHSigner(t)); !errors.Is(err, ErrRemote) {
		t.Fatalf("genuine authentication failure: %v", err)
	}
}

func TestSSHHostOfferingNoPinnedKeyTypeIsUntrusted(t *testing.T) {
	owner := testSSHSigner(t)
	ecdsaKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ecdsaHost, err := ssh.NewSignerFromKey(ecdsaKey)
	if err != nil {
		t.Fatal(err)
	}
	remote, known := serveBudgetSSHWithHost(t, owner, ecdsaHost, 0)
	client, err := NewClient(Config{AllowedRemotes: []string{remote.Address}, KnownHostsFile: known}, nil)
	if err != nil {
		t.Fatal(err)
	}
	client.allowLoopback = true
	// The genuine ECDSA pin negotiates its own type and reaches the fake pack.
	if _, err := client.Fetch(t.Context(), remote, "", owner); !errors.Is(err, ErrContent) {
		t.Fatalf("pinned ECDSA host: got %v, want ErrContent from the fake pack", err)
	}
	// An Ed25519 pin restricts negotiation to a type this host never offers,
	// so the handshake fails before the host key callback can run.
	if err := os.WriteFile(known, []byte(knownhosts.Line([]string{remote.Address}, testSSHSigner(t).PublicKey())+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, signer := range []ssh.Signer{owner, testSSHSigner(t)} {
		if _, err := client.Fetch(t.Context(), remote, "", signer); !errors.Is(err, ErrTrust) {
			t.Fatalf("host without the pinned key type: got %v, want ErrTrust", err)
		}
	}
}
