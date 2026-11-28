package auditdomain

import (
	"encoding/base64"
	"strings"
	"unicode/utf8"
)

// Outgoing bytes also form the proposal's bounded report projection. The exact
// selected-exchange artifact carries the available response body and is pinned
// through EvidenceIDs when finding intake commits.
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

// validateFindingHTTPExchange checks the exchange shape; the reference to a
// proposal evidence ID is checked by validateFindingEvidenceReferences.
func validateFindingHTTPExchange(exchange FindingHTTPExchange) error {
	if !validFindingLine(exchange.RequestID) || validateIdentifier(exchange.RequestTag, "request_tag") != nil {
		return invalid(CodeInvalid, "http_exchanges.request_id")
	}
	if len(exchange.Attempts) == 0 || len(exchange.Attempts) > MaximumHTTPExchangeAttempts {
		return invalid(CodeLimitExceeded, "http_exchanges.attempts")
	}
	for _, attempt := range exchange.Attempts {
		if err := validateFindingHTTPAttempt(attempt); err != nil {
			return err
		}
	}
	if exchange.ResponseBodyEvidenceID != "" && validateIdentifier(exchange.ResponseBodyEvidenceID, "http_exchanges.response_body_evidence_id") != nil {
		return invalid(CodeReferenceInvalid, "http_exchanges.response_body_evidence_id")
	}
	return nil
}

func validateFindingHTTPAttempt(attempt FindingHTTPAttempt) error {
	if !validFindingURL(attempt.URL) || !validFindingMethod(attempt.Method) {
		return invalid(CodeInvalid, "http_exchanges.request")
	}
	body, err := base64.StdEncoding.Strict().DecodeString(attempt.BodyBase64)
	if err != nil || base64.StdEncoding.EncodeToString(body) != attempt.BodyBase64 {
		return invalid(CodeInvalid, "http_exchanges.body_base64")
	}
	if len(body) > MaximumHTTPRequestBodyBytes {
		return invalid(CodeLimitExceeded, "http_exchanges.body_base64")
	}
	if attempt.Headers == nil || !validFindingHTTPHeaders(attempt.Headers) || !validFindingHTTPHeaders(attempt.ResponseHeaders) {
		return invalid(CodeInvalid, "http_exchanges.headers")
	}
	hasResponse, hasError := attempt.Status != nil, attempt.Error != ""
	if hasResponse == hasError {
		return invalid(CodeInvalid, "http_exchanges.outcome")
	}
	// RFC 9110 defines HTTP status codes in the range 100–599.
	if hasResponse && (*attempt.Status < 100 || *attempt.Status > 599) {
		return invalid(CodeInvalid, "http_exchanges.status")
	}
	if hasError && attempt.Error != "transport_error" && attempt.Error != "cancelled" {
		return invalid(CodeInvalid, "http_exchanges.error")
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
