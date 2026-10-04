package gitimport

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

const Deadline = 120 * time.Second
const MaxReceivedBytes = 128 << 20
const MaxDecodedBytes = 256 << 20
const maxAdvertisementBytes = 4 << 20
const maxReferences = 10000

var (
	ErrURL         = errors.New("invalid Git repository URL")
	ErrRemote      = errors.New("Git repository is unavailable")
	ErrRef         = errors.New("Git branch or tag is missing or ambiguous")
	ErrTrust       = errors.New("Git SSH host trust is unavailable or verification failed")
	ErrDestination = errors.New("Git remote destination is not allowed")
	ErrBudget      = errors.New("Git import exceeds a resource limit")
	ErrContent     = errors.New("Git snapshot contains unsupported or invalid content")
)

// UnsupportedEntryError names a tracked entry that Git import rejects rather
// than silently omitting. It matches ErrContent. The path is the repository's
// own validated, bounded tree path, returned only to the requesting owner.
type UnsupportedEntryError struct {
	Kind string
	Path string
}

func (e *UnsupportedEntryError) Error() string {
	return fmt.Sprintf("Git snapshot contains a %s at %q; Git import does not support it", e.Kind, e.Path)
}

func (e *UnsupportedEntryError) Is(target error) bool { return target == ErrContent }

type Remote struct {
	URL     string
	Scheme  string
	Address string
	User    string
	Path    string
}

var sshUserPattern = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$`)

func ParseRemote(raw string) (Remote, error) {
	if len(raw) == 0 || len(raw) > 4096 || strings.TrimSpace(raw) != raw || strings.ContainsAny(raw, "\x00\r\n\\") {
		return Remote{}, ErrURL
	}
	if strings.HasPrefix(raw, "git@") && !strings.Contains(raw, "://") {
		host, path, ok := strings.Cut(strings.TrimPrefix(raw, "git@"), ":")
		if !ok || host == "" || path == "" || strings.ContainsAny(path, "?#") {
			return Remote{}, ErrURL
		}
		// Preserve SCP's path relative to the SSH account's home. Adding only
		// a slash would silently turn git@host:repo into the absolute /repo.
		if !strings.HasPrefix(path, "/") {
			path = "/~/" + path
		}
		raw = (&url.URL{Scheme: "ssh", User: url.User("git"), Host: host, Path: path}).String()
	}
	u, err := url.Parse(raw)
	if err != nil || u.Opaque != "" || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Host == "" || u.Path == "" || u.Path == "/" || (u.Scheme != "https" && u.Scheme != "ssh") || strings.ContainsAny(u.Path, "\x00\r\n\\") {
		return Remote{}, ErrURL
	}
	user := ""
	if u.User != nil {
		if _, password := u.User.Password(); password || u.Scheme != "ssh" {
			return Remote{}, ErrURL
		}
		user = u.User.Username()
		if !sshUserPattern.MatchString(user) {
			return Remote{}, ErrURL
		}
	}
	if u.Scheme == "ssh" && user == "" {
		user = "git"
		u.User = url.User(user)
	}
	host := strings.ToLower(u.Hostname())
	port := u.Port()
	if port == "" {
		port = "443"
		if u.Scheme == "ssh" {
			port = "22"
		}
	}
	address := net.JoinHostPort(host, port)
	if ValidateConfig(Config{AllowedRemotes: []string{address}}) != nil {
		return Remote{}, ErrURL
	}
	u.Host = address
	path := u.Path
	if u.Scheme == "ssh" && strings.HasPrefix(path, "/~/") {
		path = strings.TrimPrefix(path, "/~/")
		if path == "" {
			return Remote{}, ErrURL
		}
	}
	return Remote{URL: u.String(), Scheme: u.Scheme, Address: address, User: user, Path: path}, nil
}

// Client uses go-git's protocol/pack primitives, not Clone or filesystem stores.
// Test-only fields are unexported; production always verifies actual addresses.
type Client struct {
	config        Config
	tlsConfig     *tls.Config
	allowLoopback bool
	sshWireBudget int64 // test seam; zero uses the production wire limit
	logger        *slog.Logger
}

func NewClient(cfg Config, logger *slog.Logger) (*Client, error) {
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}
	if logger == nil {
		logger = slog.Default()
	}
	cfg.AllowedRemotes = append([]string(nil), cfg.AllowedRemotes...)
	return &Client{config: cfg, logger: logger}, nil
}
func (c *Client) Allowed(remote Remote) bool {
	for _, address := range c.config.AllowedRemotes {
		if address == remote.Address {
			return true
		}
	}
	return false
}
func (c *Client) dial(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, ErrDestination
	}
	allowed := false
	for _, entry := range c.config.AllowedRemotes {
		if entry == address {
			allowed = true
		}
	}
	if !allowed {
		return nil, ErrDestination
	}
	ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil || len(ips) == 0 {
		return nil, ErrRemote
	}
	for _, entry := range ips {
		ip := entry.IP
		if (!c.allowLoopback && ip.IsLoopback()) || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast() || entry.Zone != "" {
			return nil, ErrDestination
		}
	}
	var dialer net.Dialer
	for _, entry := range ips {
		conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(entry.IP.String(), port))
		if err == nil {
			return conn, nil
		}
	}
	return nil, ErrRemote
}

type Snapshot struct {
	Data          []byte
	RepositoryURL string
	RequestedRef  *string
	Commit        string
}

func (c *Client) Fetch(ctx context.Context, remote Remote, ref string, signer ssh.Signer) (Snapshot, error) {
	ctx, cancel := context.WithTimeout(ctx, Deadline)
	defer cancel()
	canonical, err := ParseRemote(remote.URL)
	if err != nil || canonical != remote {
		return Snapshot{}, ErrURL
	}
	if !c.Allowed(remote) {
		return Snapshot{}, ErrDestination
	}
	if len(ref) > 1024 || strings.ContainsAny(ref, "\x00\r\n") {
		return Snapshot{}, ErrRef
	}
	var adv *packp.AdvRefs
	var exchange func([]byte) (io.ReadCloser, error)
	var cleanup func()
	var wire *wireBudgetConn
	if remote.Scheme == "https" {
		adv, exchange, cleanup, err = c.https(ctx, remote)
	} else if remote.Scheme == "ssh" {
		adv, exchange, cleanup, wire, err = c.ssh(ctx, remote, signer)
	} else {
		err = ErrURL
	}
	safeError := func(cause error) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if wire != nil && wire.exhausted.Load() {
			return ErrBudget
		}
		return c.safeError(ctx, remote, cause)
	}
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return Snapshot{}, safeError(err)
	}
	oid, err := resolveRef(adv, ref)
	if err != nil {
		return Snapshot{}, err
	}
	if !adv.Capabilities.Supports(capability.Shallow) {
		return Snapshot{}, ErrContent
	}
	request := packp.NewUploadPackRequest()
	request.Wants = []plumbing.Hash{oid}
	request.Depth = packp.DepthCommits(1)
	_ = request.Capabilities.Set(capability.Shallow)
	for _, cap := range []capability.Capability{capability.OFSDelta, capability.NoProgress} {
		if adv.Capabilities.Supports(cap) {
			_ = request.Capabilities.Set(cap)
		}
	}
	var body bytes.Buffer
	if err := request.UploadRequest.Encode(&body); err != nil {
		return Snapshot{}, ErrRef
	}
	body.WriteString("0009done\n")
	response, err := exchange(body.Bytes())
	if err != nil {
		return Snapshot{}, safeError(err)
	}
	defer response.Close()
	upload := packp.NewUploadPackResponse(request)
	// Limit ACK/shallow negotiation independently before pack streaming starts.
	negotiation := &boundedReader{ctx: ctx, reader: response, remaining: maxAdvertisementBytes}
	if err := upload.Decode(&countedBody{Reader: negotiation, close: response.Close}); err != nil {
		return Snapshot{}, safeError(err)
	}
	negotiation.remaining = MaxReceivedBytes
	objects, err := decodePack(ctx, upload)
	if errors.Is(err, ErrContent) {
		// The old receive step observed transport and byte-budget failures even
		// when the pack itself was malformed. Drain only that failure path with
		// a fixed buffer so a fast parser rejection does not hide either class.
		if _, drainErr := io.Copy(io.Discard, upload); drainErr != nil {
			return Snapshot{}, safeError(drainErr)
		}
	}
	if wire != nil && wire.exhausted.Load() {
		return Snapshot{}, ErrBudget
	}
	if err != nil {
		return Snapshot{}, safeError(err)
	}
	archive, commit, err := archiveSnapshot(ctx, objects, oid)
	if err != nil {
		return Snapshot{}, err
	}
	result := Snapshot{Data: archive, RepositoryURL: remote.URL, Commit: commit.String()}
	if ref != "" {
		result.RequestedRef = &ref
	}
	return result, nil
}
func resolveRef(adv *packp.AdvRefs, ref string) (plumbing.Hash, error) {
	if len(adv.References)+len(adv.Peeled) > maxReferences {
		return plumbing.ZeroHash, ErrBudget
	}
	if format := adv.Capabilities.Get(capability.Capability("object-format")); len(format) > 0 && format[0] != "sha1" {
		return plumbing.ZeroHash, ErrContent
	}
	if ref == "" {
		if adv.Head == nil || *adv.Head == plumbing.ZeroHash {
			return plumbing.ZeroHash, ErrRef
		}
		return *adv.Head, nil
	}
	if strings.HasPrefix(ref, "refs/heads/") || strings.HasPrefix(ref, "refs/tags/") {
		if plumbing.ReferenceName(ref).Validate() != nil {
			return plumbing.ZeroHash, ErrRef
		}
		if hash, ok := adv.References[ref]; ok {
			return hash, nil
		}
		return plumbing.ZeroHash, ErrRef
	}
	branch, hasBranch := adv.References["refs/heads/"+ref]
	tag, hasTag := adv.References["refs/tags/"+ref]
	if hasBranch == hasTag {
		return plumbing.ZeroHash, ErrRef
	}
	if hasBranch {
		return branch, nil
	}
	return tag, nil
}

// safeError maps a transport failure to a client-safe sentinel. The unmasked
// cause, such as a TLS verification failure, is logged only server-side.
func (c *Client) safeError(ctx context.Context, remote Remote, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	for _, safe := range []error{ErrBudget, ErrTrust, ErrDestination, ErrRef, ErrContent} {
		if errors.Is(err, safe) {
			return safe
		}
	}
	if c.logger != nil {
		// url.Error includes the request URL in Error(), including private
		// repository paths. Keep only the underlying transport cause in logs.
		logCause := err
		for {
			var requestErr *url.Error
			if !errors.As(logCause, &requestErr) {
				break
			}
			logCause = requestErr.Err
		}
		c.logger.WarnContext(ctx, "Git remote operation failed",
			"scheme", remote.Scheme, "address", remote.Address, "error", logCause)
	}
	return ErrRemote
}
func decodeAdvertisement(ctx context.Context, r io.Reader) (*packp.AdvRefs, error) {
	// The byte reader stops at the advertisement's flush (important on SSH).
	adv := packp.NewAdvRefs()
	if err := adv.Decode(&boundedReader{ctx: ctx, reader: r, remaining: maxAdvertisementBytes}); err != nil {
		return nil, err
	}
	if len(adv.References)+len(adv.Peeled) > maxReferences {
		return nil, ErrBudget
	}
	return adv, nil
}

type boundedReader struct {
	ctx       context.Context
	reader    io.Reader
	remaining int64
}

func (r *boundedReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	if r.remaining <= 0 {
		return 0, ErrBudget
	}
	if int64(len(p)) > r.remaining {
		p = p[:r.remaining]
	}
	n, err := r.reader.Read(p)
	r.remaining -= int64(n)
	return n, err
}
func (c *Client) https(ctx context.Context, remote Remote) (*packp.AdvRefs, func([]byte) (io.ReadCloser, error), func(), error) {
	transport := &http.Transport{DialContext: c.dial, TLSClientConfig: c.tlsConfig, DisableCompression: true, MaxResponseHeaderBytes: 64 << 10, ResponseHeaderTimeout: 15 * time.Second}
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	remaining := int64(MaxReceivedBytes)
	call := func(method, suffix string, body []byte) (io.ReadCloser, error) {
		req, err := http.NewRequestWithContext(ctx, method, strings.TrimSuffix(remote.URL, "/")+suffix, bytes.NewReader(body))
		if err != nil {
			return nil, ErrURL
		}
		req.Header.Set("Accept", "application/x-git-upload-pack-result")
		if method == http.MethodPost {
			req.Header.Set("Content-Type", "application/x-git-upload-pack-request")
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, fmt.Errorf("%w: HTTP status %d", ErrRemote, resp.StatusCode)
		}
		limit := &boundedReader{ctx: ctx, reader: resp.Body, remaining: remaining}
		return &countedBody{Reader: limit, close: func() error { remaining = limit.remaining; return resp.Body.Close() }}, nil
	}
	response, err := call(http.MethodGet, "/info/refs?service=git-upload-pack", nil)
	if err != nil {
		return nil, nil, transport.CloseIdleConnections, err
	}
	adv, err := decodeAdvertisement(ctx, response)
	_ = response.Close()
	return adv, func(body []byte) (io.ReadCloser, error) { return call(http.MethodPost, "/git-upload-pack", body) }, transport.CloseIdleConnections, err
}

type countedBody struct {
	io.Reader
	close func() error
}

func (r *countedBody) Close() error { return r.close() }

func pinnedHostKeyAlgorithms(verifier ssh.HostKeyCallback, host string) ([]string, error) {
	// A deliberately unusable Ed25519 key asks knownhosts which ordinary keys
	// match this host. The callback still verifies the real key after the SSH
	// handshake; this probe grants no trust to the server.
	probe, err := ssh.NewPublicKey(ed25519.PublicKey(make([]byte, ed25519.PublicKeySize)))
	if err != nil {
		return nil, ErrTrust
	}
	var mismatch *knownhosts.KeyError
	if err := verifier(host, &net.TCPAddr{}, probe); !errors.As(err, &mismatch) {
		return nil, ErrTrust
	}
	if len(mismatch.Want) == 0 {
		// An unknown host will still fail the real callback. Leaving the SSH
		// defaults also preserves hosts trusted only by a certificate authority.
		return nil, nil
	}

	pinned := make(map[string]bool, len(mismatch.Want))
	for _, known := range mismatch.Want {
		switch known.Key.Type() {
		case ssh.KeyAlgoRSA:
			pinned[ssh.KeyAlgoRSASHA512] = true
			pinned[ssh.KeyAlgoRSASHA256] = true
			pinned[ssh.KeyAlgoRSA] = true // Existing SHA-1 fallback for older servers.
		default:
			pinned[known.Key.Type()] = true
		}
	}
	algorithms := make([]string, 0, len(pinned)+8)
	supported := ssh.SupportedAlgorithms().HostKeys
	for _, algorithm := range supported {
		if pinned[algorithm] && !strings.Contains(algorithm, "-cert-v01@openssh.com") {
			algorithms = append(algorithms, algorithm)
		}
	}
	for _, algorithm := range []string{ssh.KeyAlgoRSA, ssh.InsecureKeyAlgoDSA} {
		if pinned[algorithm] {
			algorithms = append(algorithms, algorithm)
		}
	}
	if len(algorithms) == 0 {
		return nil, nil // Keep the verifier authoritative for unsupported key types.
	}
	// Certificate authorities are not included in KeyError.Want. Retain their
	// negotiated algorithms after the pinned ordinary types for mixed files.
	for _, algorithm := range append(supported, ssh.InsecureAlgorithms().HostKeys...) {
		if strings.Contains(algorithm, "-cert-v01@openssh.com") {
			algorithms = append(algorithms, algorithm)
		}
	}
	return algorithms, nil
}

func (c *Client) ssh(ctx context.Context, remote Remote, signer ssh.Signer) (*packp.AdvRefs, func([]byte) (io.ReadCloser, error), func(), *wireBudgetConn, error) {
	if signer == nil {
		return nil, nil, nil, nil, ErrRemote
	}
	if c.config.KnownHostsFile == "" {
		return nil, nil, nil, nil, ErrTrust
	}
	verifier, err := knownhosts.New(c.config.KnownHostsFile)
	if err != nil {
		return nil, nil, nil, nil, ErrTrust
	}
	hostKeyAlgorithms, err := pinnedHostKeyAlgorithms(verifier, remote.Address)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	conn, err := c.dial(ctx, "tcp", remote.Address)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	// Count the SSH wire stream as well as stdout, including a hostile stderr
	// stream and handshake/channel traffic. Exhaustion closes all channels.
	limit := int64(MaxReceivedBytes)
	if c.sshWireBudget > 0 && c.sshWireBudget < limit {
		limit = c.sshWireBudget
	}
	wire := &wireBudgetConn{Conn: conn, remaining: limit}
	conn = wire
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	cleanup := func() { stop(); _ = conn.Close() }
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	}
	cfg := &ssh.ClientConfig{User: remote.User, Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)}, HostKeyAlgorithms: hostKeyAlgorithms, HostKeyCallback: func(host string, address net.Addr, key ssh.PublicKey) error {
		if verifier(host, address, key) != nil {
			return ErrTrust
		}
		return nil
	}}
	sessionConn, channels, requests, err := ssh.NewClientConn(conn, remote.Address, cfg)
	if err != nil {
		return nil, nil, cleanup, wire, err
	}
	client := ssh.NewClient(sessionConn, channels, requests)
	session, err := client.NewSession()
	if err != nil {
		return nil, nil, cleanup, wire, err
	}
	stdout, err := session.StdoutPipe()
	if err != nil {
		return nil, nil, cleanup, wire, err
	}
	stdin, err := session.StdinPipe()
	if err != nil {
		return nil, nil, cleanup, wire, err
	}
	// Remote upload-pack is the SSH Git protocol; no local process is started.
	command := "git-upload-pack '" + strings.ReplaceAll(remote.Path, "'", "'\\''") + "'"
	if err := session.Start(command); err != nil {
		return nil, nil, cleanup, wire, err
	}
	reader := &boundedReader{ctx: ctx, reader: stdout, remaining: MaxReceivedBytes}
	adv, err := decodeAdvertisement(ctx, reader)
	exchange := func(body []byte) (io.ReadCloser, error) {
		if _, err := stdin.Write(body); err != nil {
			return nil, err
		}
		return &countedBody{Reader: reader, close: session.Close}, nil
	}
	return adv, exchange, cleanup, wire, err
}

type wireBudgetConn struct {
	net.Conn
	remaining int64
	exhausted atomic.Bool
}

func (c *wireBudgetConn) Read(p []byte) (int, error) {
	if c.remaining <= 0 {
		c.exhausted.Store(true)
		_ = c.Conn.Close()
		return 0, ErrBudget
	}
	if int64(len(p)) > c.remaining {
		p = p[:c.remaining]
	}
	n, err := c.Conn.Read(p)
	c.remaining -= int64(n)
	if c.remaining == 0 {
		c.exhausted.Store(true)
	}
	return n, err
}
