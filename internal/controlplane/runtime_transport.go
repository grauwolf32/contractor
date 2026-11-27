package controlplane

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"google.golang.org/grpc/codes"
	"io"
	"mime"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

const maxRuntimeResponseBytes = 1 << 20

var runtimePathIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

type RuntimeAPIError struct {
	RPCStatus  codes.Code
	StatusCode int
	Code       string
	Retryable  bool
}

func (e *RuntimeAPIError) Error() string {
	if e.RPCStatus != codes.OK {
		return fmt.Sprintf("Runtime Agent returned gRPC %s (%s)", e.RPCStatus, e.Code)
	}
	return fmt.Sprintf("Runtime Agent returned HTTP %d (%s)", e.StatusCode, e.Code)
}

func (c *RuntimeControlClient) doRequest(
	ctx context.Context,
	method string,
	target string,
	body []byte,
	runtimeAgentID string,
	headers map[string]string,
) (*http.Response, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, target, reader)
	if err != nil {
		return nil, errors.New("build Runtime Agent request")
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set(requestid.Header, requestid.Ensure(ctx))
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	client := c.client
	if c.transports != nil {
		transport, bindErr := c.transports.RoundTripper(runtimeAgentID)
		if bindErr != nil {
			return nil, errors.New("Runtime Agent principal binding is invalid")
		}
		client = &http.Client{
			Transport: transport,
			Timeout:   c.timeout,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return errors.New("Runtime Agent redirects are not allowed")
			},
		}
	}
	if client == nil {
		return nil, errors.New("Runtime Agent HTTP client is unavailable")
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call Runtime Agent: %w", err)
	}
	return response, nil
}

func (c *RuntimeControlClient) endpoint(baseURL, allocationID, operation string) (string, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("Runtime Agent control URL is invalid")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("Runtime Agent control URL must use HTTP or HTTPS")
	}
	if c.requireHTTPS && parsed.Scheme != "https" {
		return "", errors.New("Runtime Agent control URL must use HTTPS")
	}
	if !runtimePathIDPattern.MatchString(allocationID) {
		return "", errors.New("allocation ID is not a safe URL path segment")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/private/v1/allocations/" + allocationID + "/" + operation
	parsed.RawPath = ""
	return parsed.String(), nil
}

func decodeRuntimeError(response *http.Response) error {
	if err := requireJSONContentType(response); err != nil {
		return &RuntimeAPIError{StatusCode: response.StatusCode, Code: "invalid_error_response"}
	}
	data, err := readBoundedRuntimeBody(response.Body)
	if err != nil {
		return &RuntimeAPIError{StatusCode: response.StatusCode, Code: "invalid_error_response"}
	}
	var value privateErrorResponse
	if strictjson.Decode(data, &value) != nil || strings.TrimSpace(value.Code) == "" ||
		strings.TrimSpace(value.Message) == "" {
		return &RuntimeAPIError{StatusCode: response.StatusCode, Code: "invalid_error_response"}
	}
	return &RuntimeAPIError{
		StatusCode: response.StatusCode, Code: value.Code, Retryable: value.Retryable,
	}
}

func requireJSONContentType(response *http.Response) error {
	values := response.Header.Values("Content-Type")
	if len(values) != 1 {
		return errors.New("Runtime Agent response must have one JSON Content-Type")
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || mediaType != "application/json" || len(parameters) > 1 {
		return errors.New("Runtime Agent response Content-Type is invalid")
	}
	if len(parameters) == 1 {
		charset, ok := parameters["charset"]
		if !ok || !strings.EqualFold(charset, "utf-8") {
			return errors.New("Runtime Agent response Content-Type parameter is invalid")
		}
	}
	return nil
}

func readBoundedRuntimeBody(body io.Reader) ([]byte, error) {
	return readBoundedBody(body, maxRuntimeResponseBytes)
}

func readBoundedBody(body io.Reader, limit int) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, int64(limit)+1))
	if err != nil {
		return nil, persistencepostgres.WrapError("read Runtime Agent response", err)
	}
	if len(data) > limit {
		return nil, errors.New("Runtime Agent response is too large")
	}
	return data, nil
}
