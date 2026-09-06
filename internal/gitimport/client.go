package gitimport

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
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
const MaxArchiveBytes = 64 << 20
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
		if !ok || host == "" || path == "" {
			return Remote{}, ErrURL
		}
		raw = "ssh://git@" + host + "/" + strings.TrimPrefix(path, "/")
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
	return Remote{URL: u.String(), Scheme: u.Scheme, Address: address, User: user, Path: u.Path}, nil
}

// Client uses go-git's protocol/pack primitives, not Clone or filesystem stores.
// Test-only fields are unexported; production always verifies actual addresses.
type Client struct {
	config        Config
	tlsConfig     *tls.Config
	allowLoopback bool
}

func NewClient(cfg Config) (*Client, error) {
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}
	cfg.AllowedRemotes = append([]string(nil), cfg.AllowedRemotes...)
	return &Client{config: cfg}, nil
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
	if remote.Scheme == "https" {
		adv, exchange, cleanup, err = c.https(ctx, remote)
	} else if remote.Scheme == "ssh" {
		adv, exchange, cleanup, err = c.ssh(ctx, remote, signer)
	} else {
		err = ErrURL
	}
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		return Snapshot{}, safeError(ctx, err)
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
		return Snapshot{}, safeError(ctx, err)
	}
	defer response.Close()
	upload := packp.NewUploadPackResponse(request)
	// Limit ACK/shallow negotiation independently before pack streaming starts.
	negotiation := &boundedReader{ctx: ctx, reader: response, remaining: maxAdvertisementBytes}
	if err := upload.Decode(&countedBody{Reader: negotiation, close: response.Close}); err != nil {
		return Snapshot{}, safeError(ctx, err)
	}
	negotiation.remaining = MaxReceivedBytes
	pack, err := readBounded(ctx, upload, MaxReceivedBytes)
	if err != nil {
		return Snapshot{}, safeError(ctx, err)
	}
	objects, err := decodePack(ctx, pack)
	if err != nil {
		return Snapshot{}, err
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
func safeError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	for _, safe := range []error{ErrBudget, ErrTrust, ErrDestination, ErrRef, ErrContent} {
		if errors.Is(err, safe) {
			return safe
		}
	}
	return ErrRemote
}
func readBounded(ctx context.Context, r io.Reader, maximum int64) ([]byte, error) {
	var result bytes.Buffer
	chunk := make([]byte, 32<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := r.Read(chunk)
		if int64(result.Len()+n) > maximum {
			return nil, ErrBudget
		}
		result.Write(chunk[:n])
		if err == io.EOF {
			return result.Bytes(), nil
		}
		if err != nil {
			return nil, err
		}
	}
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
			return nil, ErrRemote
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
func (c *Client) ssh(ctx context.Context, remote Remote, signer ssh.Signer) (*packp.AdvRefs, func([]byte) (io.ReadCloser, error), func(), error) {
	if signer == nil {
		return nil, nil, nil, ErrRemote
	}
	if c.config.KnownHostsFile == "" {
		return nil, nil, nil, ErrTrust
	}
	verifier, err := knownhosts.New(c.config.KnownHostsFile)
	if err != nil {
		return nil, nil, nil, ErrTrust
	}
	conn, err := c.dial(ctx, "tcp", remote.Address)
	if err != nil {
		return nil, nil, nil, err
	}
	// Count the SSH wire stream as well as stdout, including a hostile stderr
	// stream and handshake/channel traffic. Exhaustion closes all channels.
	conn = &wireBudgetConn{Conn: conn, remaining: MaxReceivedBytes}
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	cleanup := func() { stop(); _ = conn.Close() }
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	}
	cfg := &ssh.ClientConfig{User: remote.User, Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)}, HostKeyCallback: func(host string, address net.Addr, key ssh.PublicKey) error {
		if verifier(host, address, key) != nil {
			return ErrTrust
		}
		return nil
	}}
	sessionConn, channels, requests, err := ssh.NewClientConn(conn, remote.Address, cfg)
	if err != nil {
		return nil, nil, cleanup, err
	}
	client := ssh.NewClient(sessionConn, channels, requests)
	session, err := client.NewSession()
	if err != nil {
		return nil, nil, cleanup, err
	}
	stdout, err := session.StdoutPipe()
	if err != nil {
		return nil, nil, cleanup, err
	}
	stdin, err := session.StdinPipe()
	if err != nil {
		return nil, nil, cleanup, err
	}
	// Remote upload-pack is the SSH Git protocol; no local process is started.
	command := "git-upload-pack '" + strings.ReplaceAll(remote.Path, "'", "'\\''") + "'"
	if err := session.Start(command); err != nil {
		return nil, nil, cleanup, err
	}
	reader := &boundedReader{ctx: ctx, reader: stdout, remaining: MaxReceivedBytes}
	adv, err := decodeAdvertisement(ctx, reader)
	exchange := func(body []byte) (io.ReadCloser, error) {
		if _, err := stdin.Write(body); err != nil {
			return nil, err
		}
		return &countedBody{Reader: reader, close: session.Close}, nil
	}
	return adv, exchange, cleanup, err
}

type wireBudgetConn struct {
	net.Conn
	remaining int64
}

func (c *wireBudgetConn) Read(p []byte) (int, error) {
	if c.remaining <= 0 {
		_ = c.Conn.Close()
		return 0, ErrBudget
	}
	if int64(len(p)) > c.remaining {
		p = p[:c.remaining]
	}
	n, err := c.Conn.Read(p)
	c.remaining -= int64(n)
	return n, err
}
