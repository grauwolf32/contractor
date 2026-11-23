//go:build e2e

package e2e

import (
	"context"
	"io"
	"net"
	"net/url"
	"runtime"
	"sync"
	"testing"
	"time"
)

// A local Podman VM has a different loopback namespace. Keep published APIs on
// host loopback and expose only owned fixture dependencies on the host interface.
func containerGateHost(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "linux" {
		return "127.0.0.1"
	}
	return gitGateHost(t)
}

func containerGateAddress(t *testing.T, host string) string {
	t.Helper()
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return address
}

func containerGateURL(t *testing.T, raw, host string) string {
	t.Helper()
	if runtime.GOOS == "linux" {
		return raw
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Hostname() == "" {
		t.Fatal("invalid fixture dependency URL")
	}
	ip := net.ParseIP(parsed.Hostname())
	if parsed.Hostname() != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return raw
	}
	parsed.Host = containerGateRelay(t, host, parsed.Host)
	return parsed.String()
}

func containerGateNetwork(t *testing.T, environment map[string]string, public, private, host string) []string {
	t.Helper()
	if runtime.GOOS == "linux" {
		return []string{"--network=host"}
	}
	_, publicPort, err := net.SplitHostPort(public)
	if err != nil {
		t.Fatal(err)
	}
	_, privatePort, err := net.SplitHostPort(private)
	if err != nil {
		t.Fatal(err)
	}
	environment["CONTRACTOR_PUBLIC_LISTEN"] = net.JoinHostPort("0.0.0.0", publicPort)
	environment["CONTRACTOR_PRIVATE_LISTEN"] = net.JoinHostPort("0.0.0.0", privatePort)
	environment["CONTRACTOR_DATABASE_URL"] = containerGateURL(t, environment["CONTRACTOR_DATABASE_URL"], host)
	return []string{"--network=bridge", "--publish", public + ":" + publicPort, "--publish", private + ":" + privatePort}
}

func containerGateRelay(t *testing.T, host, target string) string {
	t.Helper()
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	var workers sync.WaitGroup
	var mu sync.Mutex
	active := make(map[net.Conn]struct{})
	slots := make(chan struct{}, 64)
	workers.Add(1)
	go func() {
		defer workers.Done()
		for {
			incoming, err := listener.Accept()
			if err != nil {
				return
			}
			select {
			case slots <- struct{}{}:
			default:
				incoming.Close()
				continue
			}
			workers.Add(1)
			go func() {
				defer workers.Done()
				defer func() { <-slots }()
				defer incoming.Close()
				mu.Lock()
				if ctx.Err() != nil {
					mu.Unlock()
					return
				}
				active[incoming] = struct{}{}
				mu.Unlock()
				defer func() { mu.Lock(); delete(active, incoming); mu.Unlock() }()
				outgoing, err := (&net.Dialer{Timeout: 5 * time.Second}).DialContext(ctx, "tcp", target)
				if err != nil {
					return
				}
				defer outgoing.Close()
				mu.Lock()
				if ctx.Err() != nil {
					mu.Unlock()
					return
				}
				active[outgoing] = struct{}{}
				mu.Unlock()
				defer func() { mu.Lock(); delete(active, outgoing); mu.Unlock() }()
				deadline := time.Now().Add(6 * time.Minute)
				incoming.SetDeadline(deadline)
				outgoing.SetDeadline(deadline)
				copied := make(chan struct{})
				go func() { io.Copy(outgoing, incoming); outgoing.(*net.TCPConn).CloseWrite(); close(copied) }()
				io.Copy(incoming, outgoing)
				incoming.(*net.TCPConn).CloseWrite()
				<-copied
			}()
		}
	}()
	t.Cleanup(func() {
		cancel()
		listener.Close()
		mu.Lock()
		for connection := range active {
			connection.Close()
		}
		mu.Unlock()
		workers.Wait()
	})
	return listener.Addr().String()
}
