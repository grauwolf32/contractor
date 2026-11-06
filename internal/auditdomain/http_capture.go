package auditdomain

import (
	"encoding/base64"
	"encoding/json"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const HTTPExchangeMediaType = "application/vnd.contractor.http-exchange+json"
const HTTPExchangeSchema = "contractor.http.exchange.v1"
const MaximumHTTPResponseBodyBytes = contracts.MaxHTTPResponseBodyBytes

// CapturedHTTPExchange is normalized invocation-owned HTTP/Caido observation.
// Its publisher supplies original provenance independently of any consumer.
type CapturedHTTPExchange struct {
	Source           string                `json:"source"`
	RequestID        string                `json:"request_id"`
	RequestTag       string                `json:"request_tag,omitempty"`
	ObservedAt       string                `json:"observed_at,omitempty"`
	DeploymentMarker string                `json:"deployment_marker,omitempty"`
	Attempts         []CapturedHTTPAttempt `json:"attempts"`
}

type CapturedHTTPAttempt struct {
	Request  *CapturedHTTPRequest  `json:"request"`
	Response *CapturedHTTPResponse `json:"response,omitempty"`
	Error    string                `json:"error,omitempty"`
}

type CapturedHTTPRequest struct {
	Method  string      `json:"method"`
	URL     string      `json:"url"`
	Message HTTPMessage `json:"message"`
}

type CapturedHTTPResponse struct {
	Status   int         `json:"status"`
	Message  HTTPMessage `json:"message"`
	Complete bool        `json:"complete"`
}

// HTTPMessage preserves ordered headers and body bytes for the direct client,
// or the exact raw Caido message without decoding or rewriting its framing.
type HTTPMessage struct {
	Kind       string               `json:"kind"`
	Headers    *[]FindingHTTPHeader `json:"headers,omitempty"`
	BodyBase64 *string              `json:"body_base64,omitempty"`
	DataBase64 *string              `json:"data_base64,omitempty"`
}

func (capture *CapturedHTTPExchange) UnmarshalJSON(data []byte) error {
	type plain CapturedHTTPExchange
	var value plain
	// Capture has its own byte bounds: base64 response bodies may exceed the
	// ordinary Audit document/string bounds. Keep the same strict member,
	// Unicode, depth and number checks without silently shrinking FND-R53.
	if len(data) > contracts.MaxArtifactPayloadBytes || !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) || strictjson.HasNULEscape(data) {
		return invalid(CodeInvalid, "httpEvidence")
	}
	raw, err := strictjson.DecodeBounded(data, strictjson.Bounds{MaxDepth: MaximumJSONDepth, MaxNodes: MaximumJSONNodes, MaxStringBytes: base64.StdEncoding.EncodedLen(MaximumHTTPResponseBodyBytes)}, finiteJSONNumber)
	if err != nil {
		return invalid(CodeInvalid, "httpEvidence")
	}
	normalized, err := json.Marshal(raw)
	if err != nil || strictjson.DecodeNonNull(normalized, &value) != nil {
		return invalid(CodeInvalid, "httpEvidence")
	}
	if err := validateFindingJSONFields(raw, "source request_id attempts", "request_tag observed_at deployment_marker", ""); err != nil {
		return err
	}
	fields := raw.(map[string]any)
	if tag, present := fields["request_tag"]; present && tag == "" {
		return invalid(CodeInvalid, "httpEvidence.request_tag")
	}
	for _, field := range []string{"observed_at", "deployment_marker"} {
		if value, present := fields[field]; present && value == "" {
			return invalid(CodeInvalid, "httpEvidence."+field)
		}
	}
	attempts, ok := fields["attempts"].([]any)
	if !ok {
		return invalid(CodeInvalid, "httpEvidence.attempts")
	}
	for _, item := range attempts {
		if err := validateCapturedAttemptJSON(item); err != nil {
			return err
		}
	}
	*capture = CapturedHTTPExchange(value)
	return ValidateCapturedHTTPExchange(*capture)
}

func validateCapturedAttemptJSON(value any) error {
	if err := validateFindingJSONFields(value, "request", "response error", ""); err != nil {
		return err
	}
	fields := value.(map[string]any)
	if outcome, present := fields["error"]; present && outcome == "" {
		return invalid(CodeInvalid, "httpEvidence.error")
	}
	for _, name := range []string{"request", "response"} {
		message, present := fields[name]
		if !present {
			continue
		}
		required := "method url message"
		if name == "response" {
			required = "status message complete"
		}
		if err := validateFindingJSONFields(message, required, "", ""); err != nil {
			return err
		}
		body := message.(map[string]any)["message"]
		object, ok := body.(map[string]any)
		if !ok {
			return invalid(CodeInvalid, "httpEvidence.message")
		}
		switch object["kind"] {
		case "raw":
			if err := validateFindingJSONFields(body, "kind data_base64", "", ""); err != nil {
				return err
			}
		case "structured":
			if err := validateFindingJSONFields(body, "kind headers body_base64", "", ""); err != nil {
				return err
			}
			headers, ok := object["headers"].([]any)
			if !ok {
				return invalid(CodeInvalid, "httpEvidence.headers")
			}
			for _, header := range headers {
				if err := validateFindingJSONFields(header, "name value", "", ""); err != nil {
					return err
				}
			}
		default:
			return invalid(CodeInvalid, "httpEvidence.message.kind")
		}
	}
	return nil
}

func ValidateCapturedHTTPExchange(value CapturedHTTPExchange) error {
	if (value.Source != "http" && value.Source != "caido") || value.RequestID == "" ||
		len(value.RequestID) > 256 || !utf8.ValidString(value.RequestID) ||
		strings.IndexFunc(value.RequestID, func(r rune) bool { return unicode.IsControl(r) && r < 32 }) >= 0 {
		return invalid(CodeInvalid, "httpEvidence.source")
	}
	if value.RequestTag != "" && validateIdentifier(value.RequestTag, "httpEvidence.request_tag") != nil {
		return invalid(CodeInvalid, "httpEvidence.request_tag")
	}
	if value.Source == "http" {
		id, err := strconv.ParseInt(value.RequestID, 10, 64)
		if err != nil || !validFindingLine(id) || strconv.FormatInt(id, 10) != value.RequestID ||
			validateIdentifier(value.RequestTag, "httpEvidence.request_tag") != nil {
			return invalid(CodeInvalid, "httpEvidence.request_id")
		}
	}
	if value.ObservedAt != "" {
		parsed, err := time.Parse(time.RFC3339Nano, value.ObservedAt)
		if err != nil || parsed.Year() < 1 || parsed.UTC().Format(time.RFC3339Nano) != value.ObservedAt {
			return invalid(CodeInvalid, "httpEvidence.observed_at")
		}
	}
	if value.DeploymentMarker != "" && (len(value.DeploymentMarker) > 256 || strings.IndexFunc(value.DeploymentMarker, func(r rune) bool { return r < 32 }) >= 0) {
		return invalid(CodeInvalid, "httpEvidence.deployment_marker")
	}

	if len(value.Attempts) < 1 || len(value.Attempts) > MaximumHTTPExchangeAttempts {
		return invalid(CodeLimitExceeded, "httpEvidence.attempts")
	}
	for _, attempt := range value.Attempts {
		if attempt.Request == nil || !validFindingMethod(attempt.Request.Method) || !validFindingURL(attempt.Request.URL) {
			return invalid(CodeInvalid, "httpEvidence.request")
		}
		if err := validateHTTPMessage(attempt.Request.Message, MaximumHTTPRequestBodyBytes); err != nil {
			return err
		}
		if attempt.Request.Message.Kind == "raw" && *attempt.Request.Message.DataBase64 == "" {
			return invalid(CodeInvalid, "httpEvidence.request")
		}
		expectedKind := "structured"
		if value.Source == "caido" {
			expectedKind = "raw"
		}
		if attempt.Request.Message.Kind != expectedKind || attempt.Response != nil && attempt.Response.Message.Kind != expectedKind {
			return invalid(CodeInvalid, "httpEvidence.message.source")
		}
		if attempt.Response == nil && attempt.Error == "" ||
			attempt.Response != nil && attempt.Response.Complete && attempt.Error != "" {
			return invalid(CodeInvalid, "httpEvidence.outcome")
		}
		if attempt.Error != "" && attempt.Error != "transport_error" && attempt.Error != "timeout" &&
			attempt.Error != "cancelled" && attempt.Error != "response_invalid" && attempt.Error != "response_too_large" &&
			attempt.Error != "target_denied" && attempt.Error != "request_invalid" && attempt.Error != "response_unavailable" {
			return invalid(CodeInvalid, "httpEvidence.error")
		}
		if response := attempt.Response; response != nil {
			if response.Status < 100 || response.Status > 599 {
				return invalid(CodeInvalid, "httpEvidence.status")
			}
			if err := validateHTTPMessage(response.Message, MaximumHTTPResponseBodyBytes); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateHTTPMessage(value HTTPMessage, maximum int) error {
	var encoded *string
	switch value.Kind {
	case "structured":
		if value.Headers == nil || value.BodyBase64 == nil || value.DataBase64 != nil || !validFindingHTTPHeaders(*value.Headers) {
			return invalid(CodeInvalid, "httpEvidence.message")
		}
		encoded = value.BodyBase64
	case "raw":
		if value.DataBase64 == nil || value.Headers != nil || value.BodyBase64 != nil {
			return invalid(CodeInvalid, "httpEvidence.message")
		}
		encoded = value.DataBase64
	default:
		return invalid(CodeInvalid, "httpEvidence.message.kind")
	}
	if len(*encoded) > base64.StdEncoding.EncodedLen(maximum) {
		return invalid(CodeLimitExceeded, "httpEvidence.body")
	}
	data, err := base64.StdEncoding.Strict().DecodeString(*encoded)
	if err != nil || len(data) > maximum || base64.StdEncoding.EncodeToString(data) != *encoded {
		return invalid(CodeInvalid, "httpEvidence.body")
	}
	return nil
}

// HTTPExchangeOrigin is historical provenance supplied by the trusted publisher.
type HTTPExchangeOrigin struct {
	RunID        string `json:"run_id" strictjson:"required"`
	InvocationID string `json:"invocation_id" strictjson:"required"`
}

// HTTPExchangeArtifact is consumer-independent selected HTTP evidence.
type HTTPExchangeArtifact struct {
	Schema  string               `json:"schema" strictjson:"required"`
	Origin  HTTPExchangeOrigin   `json:"origin" strictjson:"required"`
	Capture CapturedHTTPExchange `json:"capture" strictjson:"required"`
}

// DecodeHTTPExchangeArtifact strictly validates one retained capture envelope.
func DecodeHTTPExchangeArtifact(data []byte) (HTTPExchangeArtifact, error) {
	var value HTTPExchangeArtifact
	if len(data) > contracts.MaxArtifactPayloadBytes || strictjson.DecodeNonNull(data, &value) != nil || value.Schema != HTTPExchangeSchema || validateIdentifier(value.Origin.RunID, "origin.run_id") != nil || validateIdentifier(value.Origin.InvocationID, "origin.invocation_id") != nil || ValidateCapturedHTTPExchange(value.Capture) != nil {
		return value, invalid(CodeInvalid, "httpEvidence")
	}
	if value.Capture.Source == "http" && value.Capture.ObservedAt == "" {
		return value, invalid(CodeInvalid, "httpEvidence.observed_at")
	}
	return value, nil
}

// HTTPExchangeArtifactName supplies one natural identity per invocation capture.
func HTTPExchangeArtifactName(invocationID, source, requestID string) string {
	data, _ := strictjson.Canonical([]string{invocationID, source, requestID})
	return "exchange-" + strings.TrimPrefix(contentdigest.Bytes(data), "sha256:")
}

// EncodeHTTPExchangeArtifact preserves the original capture and trusted origin.
func EncodeHTTPExchangeArtifact(value CapturedHTTPExchange, runID, invocationID string) ([]byte, error) {
	data, err := strictjson.Canonical(HTTPExchangeArtifact{HTTPExchangeSchema, HTTPExchangeOrigin{runID, invocationID}, value})
	if err != nil {
		return nil, err
	}
	if _, err = DecodeHTTPExchangeArtifact(data); err != nil {
		return nil, err
	}
	return data, nil
}
