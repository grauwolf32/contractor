package mtls

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// RuntimeChannels owns bounded native gRPC channels. Borrowed channels retire
// without interrupting sibling allocations; the last borrower closes them.
type RuntimeChannels struct {
	mu        sync.Mutex
	base      *tls.Config
	entries   map[string]*rpcEndpoint
	closed    bool
	now       func() time.Time
	afterFunc func(time.Duration, func()) *time.Timer
}
type rpcEndpoint struct {
	principal   string
	current     *rpcGeneration
	generations map[*rpcGeneration]struct{}
	active      int
}
type rpcGeneration struct {
	channel       *grpc.ClientConn
	created, used time.Time
	active        int
	retired       bool
	idle          *time.Timer
	peerRejected  *atomic.Bool
}

func (*RuntimeChannels) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "mtls.RuntimeChannels([REDACTED])")
}
func NewRuntimeChannels(base *tls.Config) (*RuntimeChannels, error) {
	if base == nil {
		return nil, errors.New("Runtime TLS configuration required")
	}
	clone := base.Clone()
	clone.ClientSessionCache = nil
	return &RuntimeChannels{base: clone, entries: map[string]*rpcEndpoint{}, now: time.Now, afterFunc: time.AfterFunc}, nil
}

// GRPCAuthority accepts only an HTTPS origin. Paths cannot silently disappear
// when an advertised control endpoint becomes a native gRPC authority.
func GRPCAuthority(origin string) (string, error) {
	parsed, err := contracts.ParsePrivateRPCOrigin("gRPC endpoint", origin)
	if err != nil {
		return "", err
	}

	port := parsed.Port()
	if port == "" {
		port = "443"
	}
	return net.JoinHostPort(parsed.Hostname(), port), nil
}

func (o *RuntimeChannels) Borrow(ctx context.Context, origin, principal string) (*grpc.ClientConn, func(), error) {
	authority, err := GRPCAuthority(origin)
	if err != nil {
		return nil, nil, err
	}
	bound, err := BindRuntimeAgentPrincipal(o.base, principal)
	if err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return nil, nil, ErrRuntimeChannelsClosed
	}
	key := principal + "@" + authority
	now := o.now()
	for k, e := range o.entries {
		if e.active == 0 && (e.current == nil || now.Sub(e.current.used) >= privateConnectionIdleReuse) {
			o.retire(e)
			delete(o.entries, k)
		}
	}
	e := o.entries[key]
	if e == nil {
		if len(o.entries) >= runtimeAgentTransportCapacity {
			var oldestKey string
			var oldest *rpcEndpoint
			for candidateKey, candidate := range o.entries {
				if candidate.active == 0 && (oldest == nil || candidate.current.used.Before(oldest.current.used)) {
					oldestKey, oldest = candidateKey, candidate
				}
			}
			if oldest == nil {
				return nil, nil, status.Error(codes.ResourceExhausted, "runtime_channel_capacity")
			}
			o.retire(oldest)
			delete(o.entries, oldestKey)
		}
		e = &rpcEndpoint{principal: principal, generations: map[*rpcGeneration]struct{}{}}
		o.entries[key] = e
	}
	if e.active >= privateConnectionsPerPeer {
		return nil, nil, status.Error(codes.ResourceExhausted, "runtime_rpc_capacity")
	}
	if g := e.current; g != nil && (now.Sub(g.created) >= privateConnectionReuseLifetime || now.Sub(g.used) >= privateConnectionIdleReuse) {
		o.retire(e)
	}
	if e.current == nil {
		rejected := new(atomic.Bool)
		conn, err := grpc.NewClient(authority, grpc.WithTransportCredentials(&verifiedRPCCredentials{TransportCredentials: credentials.NewTLS(bound), rejected: rejected}), grpc.WithDisableRetry(), grpc.WithMaxHeaderListSize(privateRPCMetadataBytes), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(1<<20), grpc.MaxCallSendMsgSize(1<<20)))
		if err != nil {
			return nil, nil, err
		}
		e.current = &rpcGeneration{channel: conn, created: now, used: now, peerRejected: rejected}
		e.generations[e.current] = struct{}{}
	}
	g := e.current
	if g.idle != nil {
		g.idle.Stop()
		g.idle = nil
	}
	g.active++
	e.active++
	var once sync.Once
	release := func() {
		once.Do(func() {
			o.mu.Lock()
			defer o.mu.Unlock()
			g.active--
			e.active--
			g.used = o.now()
			if g.retired && g.active == 0 {
				_ = g.channel.Close()
				delete(e.generations, g)
			} else if !g.retired && g.active == 0 {
				g.idle = o.afterFunc(privateConnectionIdleReuse, func() { o.expireIdle(key, e, g) })
			}
			if e.current == nil && e.active == 0 {
				delete(o.entries, key)
			}
		})
	}
	return g.channel, release, nil
}
func (o *RuntimeChannels) retire(e *rpcEndpoint) {
	if g := e.current; g != nil {
		if g.idle != nil {
			g.idle.Stop()
			g.idle = nil
		}
		g.retired = true
		e.current = nil
		if g.active == 0 {
			_ = g.channel.Close()
			delete(e.generations, g)
		}
	}
}
func (o *RuntimeChannels) Evict(principal string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	for key, e := range o.entries {
		if e.principal == principal {
			o.retire(e)
			if e.active == 0 {
				delete(o.entries, key)
			}
		}
	}
}
func (o *RuntimeChannels) Close() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.closed = true
	for key, e := range o.entries {
		o.retire(e)
		if e.active == 0 {
			delete(o.entries, key)
		}
	}
}

// expireIdle runs under the owner's lock; a timer from a previous idle period
// cannot retire a borrowed generation or a channel used since it was armed.
func (o *RuntimeChannels) expireIdle(key string, e *rpcEndpoint, g *rpcGeneration) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if e.current != g || g.active != 0 || o.now().Sub(g.used) < privateConnectionIdleReuse {
		return
	}
	o.retire(e)
	if e.active == 0 {
		delete(o.entries, key)
	}
}

// PeerRejected distinguishes certificate/principal rejection from an unavailable
// endpoint without parsing grpc-go's rendered transport error or exposing it.
func (o *RuntimeChannels) PeerRejected(channel *grpc.ClientConn) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	for _, endpoint := range o.entries {
		for generation := range endpoint.generations {
			if generation.channel == channel {
				return generation.peerRejected.Load()
			}
		}
	}
	return false
}

type verifiedRPCCredentials struct {
	credentials.TransportCredentials
	rejected *atomic.Bool
}

func (c *verifiedRPCCredentials) ClientHandshake(ctx context.Context, authority string, connection net.Conn) (net.Conn, credentials.AuthInfo, error) {
	secured, info, err := c.TransportCredentials.ClientHandshake(ctx, authority, connection)
	var verification *tls.CertificateVerificationError
	var unknown x509.UnknownAuthorityError
	var invalid x509.CertificateInvalidError
	var hostname x509.HostnameError
	if errors.Is(err, ErrRuntimeAgentIdentity) || errors.As(err, &verification) || errors.As(err, &unknown) || errors.As(err, &invalid) || errors.As(err, &hostname) {
		c.rejected.Store(true)
	}
	return secured, info, err
}
func (c *verifiedRPCCredentials) Clone() credentials.TransportCredentials {
	return &verifiedRPCCredentials{TransportCredentials: c.TransportCredentials.Clone(), rejected: c.rejected}
}
