package auditdomain

import (
	"encoding/base64"
	"slices"
	"strings"
	"unicode/utf8"
)

// Outgoing bytes belong to the retained proposal. An existing response body is
// pinned through EvidenceIDs, so no new per-request artifact is necessary.
type FindingHTTPExchange struct {
	RequestID              int64                `json:"request_id"`
	RequestTag             string               `json:"request_tag"`
	Attempts               []FindingHTTPAttempt `json:"attempts"`
	ResponseBodyEvidenceID string               `json:"response_body_evidence_id,omitempty"`
}

type FindingHTTPHeader struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type FindingHTTPAttempt struct {
	Method          string              `json:"method"`
	URL             string              `json:"url"`
	Headers         []FindingHTTPHeader `json:"headers"`
	BodyBase64      string              `json:"body_base64"`
	Status          *int                `json:"status,omitempty"`
	ResponseHeaders []FindingHTTPHeader `json:"response_headers,omitempty"`
	Error           string              `json:"error,omitempty"`
}

func validateFindingHTTPExchange(exchange FindingHTTPExchange, evidenceIDs []string) error {
	if !validFindingLine(exchange.RequestID) || validateIdentifier(exchange.RequestTag, "request_tag") != nil {
		return invalid(CodeInvalid, "http_exchange.request_id")
	}
	if len(exchange.Attempts) == 0 || len(exchange.Attempts) > MaximumHTTPExchangeAttempts {
		return invalid(CodeLimitExceeded, "http_exchange.attempts")
	}
	for _, attempt := range exchange.Attempts {
		if err := validateFindingHTTPAttempt(attempt); err != nil {
			return err
		}
	}
	if exchange.ResponseBodyEvidenceID != "" && !slices.Contains(evidenceIDs, exchange.ResponseBodyEvidenceID) {
		return invalid(CodeReferenceInvalid, "http_exchange.response_body_evidence_id")
	}
	return nil
}

func validateFindingHTTPAttempt(attempt FindingHTTPAttempt) error {
	if !validFindingURL(attempt.URL) || !validFindingMethod(attempt.Method) {
		return invalid(CodeInvalid, "http_exchange.request")
	}
	body, err := base64.StdEncoding.Strict().DecodeString(attempt.BodyBase64)
	if err != nil || base64.StdEncoding.EncodeToString(body) != attempt.BodyBase64 {
		return invalid(CodeInvalid, "http_exchange.body_base64")
	}
	if len(body) > MaximumHTTPRequestBodyBytes {
		return invalid(CodeLimitExceeded, "http_exchange.body_base64")
	}
	if attempt.Headers == nil || !validFindingHTTPHeaders(attempt.Headers) || !validFindingHTTPHeaders(attempt.ResponseHeaders) {
		return invalid(CodeInvalid, "http_exchange.headers")
	}
	hasResponse, hasError := attempt.Status != nil, attempt.Error != ""
	if hasResponse == hasError {
		return invalid(CodeInvalid, "http_exchange.outcome")
	}
	// RFC 9110 defines HTTP status codes in the range 100–599.
	if hasResponse && (*attempt.Status < 100 || *attempt.Status > 599) {
		return invalid(CodeInvalid, "http_exchange.status")
	}
	if hasError && attempt.Error != "transport_error" && attempt.Error != "cancelled" {
		return invalid(CodeInvalid, "http_exchange.error")
	}
	return nil
}

func validFindingHTTPHeaders(headers []FindingHTTPHeader) bool {
	byteCount := 0
	for _, header := range headers {
		byteCount += len(header.Name) + len(header.Value)
		if !httpToken.MatchString(header.Name) || strings.ContainsAny(header.Value, "\r\n\x00") || !utf8.ValidString(header.Value) {
			return false
		}
	}
	return byteCount <= MaximumHTTPHeaderBytes
}
