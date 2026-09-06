package publicclient

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
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

const (
	APIVersion        = "contractor.public.v1"
	APIVersionHeader  = "X-Contractor-API-Version"
	maximumErrorBytes = 64 * 1024
	maximumTokenBytes = 8 * 1024
)

type Options struct {
	Server    string
	Token     string
	CAFile    string
	AllowHTTP bool
	Timeout   time.Duration
	UserAgent string
}

type Client struct {
	API    *publicapi.ClientWithResponses
	origin string
}

func New(options Options) (*Client, error) {
	origin, err := NormalizeServer(options.Server, options.AllowHTTP)
	if err != nil {
		return nil, err
	}
	if err := validateToken(options.Token); err != nil {
		return nil, err
	}
	if options.Timeout <= 0 {
		return nil, errors.New("request timeout must be positive")
	}
	transport, err := transport(options.CAFile)
	if err != nil {
		return nil, err
	}
	roundTripper := &checkedTransport{
		base:      transport,
		origin:    origin,
		token:     options.Token,
		userAgent: options.UserAgent,
	}
	httpClient := &http.Client{
		Transport: roundTripper,
		Timeout:   options.Timeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return errors.New("Contractor public API redirects are not allowed")
		},
	}
	generated, err := publicapi.NewClientWithResponses(origin, publicapi.WithHTTPClient(httpClient))
	if err != nil {
		return nil, fmt.Errorf("create generated public API client: %w", err)
	}
	return &Client{API: generated, origin: origin}, nil
}

func (c *Client) Origin() string { return c.origin }

func NormalizeServer(value string, allowHTTP bool) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(value))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || parsed.User != nil ||
		parsed.RawQuery != "" || parsed.Fragment != "" || (parsed.Path != "" && parsed.Path != "/") {
		return "", errors.New("server must be an absolute origin without credentials, query, fragment, or path")
	}
	switch parsed.Scheme {
	case "https":
	case "http":
		host := net.ParseIP(parsed.Hostname())
		if !allowHTTP && (host == nil || !host.IsLoopback()) {
			return "", errors.New("HTTP is allowed only for an IP-literal loopback server; use --allow-http explicitly otherwise")
		}
	default:
		return "", errors.New("server scheme must be https or http")
	}
	parsed.Path = ""
	return strings.TrimSuffix(parsed.String(), "/"), nil
}

func transport(caFile string) (*http.Transport, error) {
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		return nil, errors.New("default HTTP transport is unavailable")
	}
	result := base.Clone()
	if strings.TrimSpace(caFile) == "" {
		return result, nil
	}
	pem, err := os.ReadFile(filepath.Clean(caFile))
	if err != nil {
		return nil, fmt.Errorf("read CA file: %w", err)
	}
	roots, err := x509.SystemCertPool()
	if err != nil || roots == nil {
		roots = x509.NewCertPool()
	}
	if !roots.AppendCertsFromPEM(pem) {
		return nil, errors.New("CA file contains no readable certificate")
	}
	result.TLSClientConfig = &tls.Config{RootCAs: roots, MinVersion: tls.VersionTLS12}
	return result, nil
}

type checkedTransport struct {
	base      http.RoundTripper
	origin    string
	token     string
	userAgent string
}

func (t *checkedTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL == nil || request.URL.Scheme+"://"+request.URL.Host != t.origin ||
		!strings.HasPrefix(request.URL.EscapedPath(), "/v1/") {
		return nil, errors.New("public API request escaped the configured Server boundary")
	}
	request.Header.Set("Authorization", "Bearer "+t.token)
	request.Header.Set("Accept", "application/json")
	if t.userAgent != "" {
		request.Header.Set("User-Agent", t.userAgent)
	}
	response, err := t.base.RoundTrip(request)
	if err != nil {
		return nil, err
	}
	versions := response.Header.Values(APIVersionHeader)
	if len(versions) != 1 || strings.TrimSpace(versions[0]) != APIVersion {
		_ = response.Body.Close()
		actual := "missing"
		if len(versions) != 0 {
			actual = strings.Join(versions, ",")
		}
		return nil, &CompatibilityError{Expected: APIVersion, Actual: actual}
	}
	if response.StatusCode < http.StatusBadRequest || response.Body == nil {
		return response, nil
	}
	body, readErr := io.ReadAll(io.LimitReader(response.Body, maximumErrorBytes+1))
	_ = response.Body.Close()
	if readErr != nil {
		return nil, fmt.Errorf("read bounded public API error: %w", readErr)
	}
	if len(body) > maximumErrorBytes {
		body, _ = json.Marshal(errorEnvelope{
			Code:      "invalid_error_response",
			Message:   "Server error response exceeded the client limit",
			Retryable: false,
			RequestID: response.Header.Get("X-Request-ID"),
		})
	}
	response.Body = io.NopCloser(bytes.NewReader(body))
	response.ContentLength = int64(len(body))
	response.Header.Set("Content-Length", fmt.Sprint(len(body)))
	return response, nil
}

type CompatibilityError struct {
	Expected string
	Actual   string
}

func (e *CompatibilityError) Error() string {
	return fmt.Sprintf("incompatible Contractor API version: expected %s, received %s", e.Expected, e.Actual)
}

type response interface {
	StatusCode() int
	GetBody() []byte
}

func CheckResponse(value response, expected ...int) error {
	if value == nil {
		return errors.New("public API returned no response")
	}
	status := value.StatusCode()
	for _, candidate := range expected {
		if status == candidate {
			return nil
		}
	}
	return DecodeError(status, value.GetBody())
}

type APIError struct {
	Status    int
	Code      string
	Message   string
	Retryable bool
	RequestID string
}

func (e *APIError) Error() string {
	detail := fmt.Sprintf("Contractor API: %s", e.Message)
	if e.Code != "" {
		detail += " (" + e.Code + ")"
	}
	if e.RequestID != "" {
		detail += " [request " + e.RequestID + "]"
	}
	return detail
}

type errorEnvelope struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"requestId"`
}

func DecodeError(status int, body []byte) error {
	var envelope errorEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Message == "" {
		return &APIError{Status: status, Code: "invalid_error_response", Message: http.StatusText(status)}
	}
	return &APIError{
		Status: status, Code: envelope.Code, Message: envelope.Message,
		Retryable: envelope.Retryable, RequestID: envelope.RequestID,
	}
}

func NewIdempotencyKey() (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", fmt.Errorf("generate idempotency key: %w", err)
	}
	return "cli_" + hex.EncodeToString(buffer), nil
}

func QuoteETag(revision string) (string, error) {
	if revision == "" || len(revision) > 256 || strings.ContainsAny(revision, "\"\\\r\n") {
		return "", errors.New("revision cannot be represented as a strong ETag")
	}
	return `"` + revision + `"`, nil
}

func ReadTokenFile(path string) (string, error) {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return "", fmt.Errorf("open API token file: %w", err)
	}
	defer file.Close()
	value, err := io.ReadAll(io.LimitReader(file, maximumTokenBytes+1))
	if err != nil {
		return "", fmt.Errorf("read API token file: %w", err)
	}
	if len(value) > maximumTokenBytes {
		return "", errors.New("API token file exceeds 8 KiB")
	}
	value = bytes.TrimSuffix(value, []byte("\n"))
	value = bytes.TrimSuffix(value, []byte("\r"))
	token := string(value)
	if err := validateToken(token); err != nil {
		return "", err
	}
	return token, nil
}

func validateToken(token string) error {
	if token == "" || len(token) > maximumTokenBytes || strings.ContainsAny(token, "\x00\r\n") {
		return errors.New("API token must contain 1 through 8192 bytes without NUL or newlines")
	}
	return nil
}

func ContextWithTimeout(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(parent, timeout)
}
