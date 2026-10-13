package auditdomain

import (
	"encoding/base64"
	"encoding/json"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const HTTPExchangeMediaType = "application/vnd.contractor.http-exchange+json"
const HTTPExchangeSchema = "contractor.http.exchange.v1"
const MaximumHTTPResponseBodyBytes = 16 << 20

// CapturedHTTPExchange is trusted Runtime capture submitted with a finding.
// It has no ArtifactRef: intake publishes it and its finding atomically.
type CapturedHTTPExchange struct {
	Source     string                `json:"source"`
	RequestID  string                `json:"request_id"`
	RequestTag string                `json:"request_tag,omitempty"`
	Attempts   []CapturedHTTPAttempt `json:"attempts"`
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
	raw, err := decodeStrictJSON(data, &value)
	if err != nil {
		return err
	}
	if err := validateFindingJSONFields(raw, "source request_id attempts", "request_tag", ""); err != nil {
		return err
	}
	fields := raw.(map[string]any)
	if tag, present := fields["request_tag"]; present && tag == "" {
		return invalid(CodeInvalid, "httpEvidence.request_tag")
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
			attempt.Error != "target_denied" && attempt.Error != "request_invalid" {
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

// EncodeHTTPExchangeArtifact binds validated capture to its Server-issued
// finding identities. Runtime cannot supply or override this association.
func EncodeHTTPExchangeArtifact(value CapturedHTTPExchange, proposalID, receiptID string) ([]byte, error) {
	if err := ValidateCapturedHTTPExchange(value); err != nil {
		return nil, err
	}
	if validateIdentifier(proposalID, "proposal_id") != nil || validateIdentifier(receiptID, "receipt_id") != nil {
		return nil, invalid(CodeInvalid, "httpEvidence.finding")
	}
	type findingAssociation struct {
		ProposalID string `json:"proposal_id"`
		ReceiptID  string `json:"receipt_id"`
	}
	return json.Marshal(struct {
		Schema  string               `json:"schema"`
		Capture CapturedHTTPExchange `json:"capture"`
		Finding findingAssociation   `json:"finding"`
	}{Schema: HTTPExchangeSchema, Capture: value,
		Finding: findingAssociation{proposalID, receiptID},
	})
}
