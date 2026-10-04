package gatewayrecovery

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Failure stores a safe classification, never provider response content.
type Failure struct {
	Code      string
	Retryable bool
	// Abandoned marks a request the Gateway received but did not answer
	// before the client deadline. The failure is transient, yet the Gateway
	// may still be generating the answer (an OpenAI-compatible proxy does not
	// cancel upstream inference when its client disconnects), so resending
	// the request would queue duplicate inference behind the abandoned one.
	Abandoned bool
}

const maximumClassificationBytes = 16 * 1024

// LiteLLM wraps the upstream OpenAI-compatible error as a Python dict repr.
// Only this observed wrapper is recognized; the wrapped text itself still has
// to be declared as a signature.
const (
	litellmWrapperPrefix = "litellm.BadRequestError: OpenAIException - Error code: 400 - "
	litellmWrapperSuffix = ". Received Model Group="
)

// Classify applies the openai-compatible@1 status rules plus the Gateway's
// declared failure signatures. Signature text is matched exactly so arbitrary
// 4xx bodies can never become a transient availability failure.
func Classify(status int, header http.Header, body []byte, signatures contracts.GatewayFailureSignatures) Failure {
	var code, message string
	var response map[string]json.RawMessage
	if len(body) <= maximumClassificationBytes && json.Unmarshal(body, &response) == nil {
		selected := json.RawMessage(body)
		if wrapped, present := response["error"]; present {
			selected = wrapped
		}
		var detail map[string]json.RawMessage
		if json.Unmarshal(selected, &detail) == nil {
			// A provider may encode code as a number. Its type must not discard
			// a valid message signature (or vice versa).
			_ = json.Unmarshal(detail["code"], &code)
			_ = json.Unmarshal(detail["message"], &message)
		} else {
			_ = json.Unmarshal(selected, &message)
		}
	}
	for _, permanentCode := range signatures.PermanentCodes {
		if code == permanentCode {
			return Failure{Code: permanentCode, Retryable: false}
		}
	}
	if modelUnavailable(status, message, signatures) {
		return Failure{Code: "model_unavailable", Retryable: true}
	}
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return Failure{Code: "gateway_access_denied", Retryable: false}
	case http.StatusTooManyRequests:
		return Failure{Code: "gateway_rate_limited", Retryable: true}
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return Failure{Code: "gateway_timeout", Retryable: true}
	case http.StatusConflict:
		return Failure{Code: "gateway_unavailable", Retryable: true}
	}
	if status >= 500 && status < 600 || header.Get("x-should-retry") == "true" {
		return Failure{Code: "gateway_unavailable", Retryable: true}
	}
	return Failure{Code: "gateway_request_rejected", Retryable: false}
}

func modelUnavailable(status int, message string, signatures contracts.GatewayFailureSignatures) bool {
	for _, signature := range signatures.ModelUnavailable {
		if signature.Status != status {
			continue
		}
		if signature.MessageEquals != "" && message == signature.MessageEquals {
			return true
		}
		if signature.LiteLLMWrapped != "" && litellmWrapped(message, signature.LiteLLMWrapped) {
			return true
		}
	}
	return false
}

// litellmWrapped compares against Python's repr of {"error": text}, which is
// how LiteLLM renders the upstream body; single quotes unless the text itself
// contains one and no double quote.
func litellmWrapped(message, upstreamText string) bool {
	if !strings.HasPrefix(message, litellmWrapperPrefix) {
		return false
	}
	upstream := strings.TrimPrefix(message, litellmWrapperPrefix)
	wrapped := "{'error': " + pythonRepr(upstreamText) + "}"
	return upstream == wrapped || strings.HasPrefix(upstream, wrapped+litellmWrapperSuffix)
}

// pythonRepr mirrors CPython's str repr quoting for the printable text that
// signature validation admits: no control characters, so no escaping beyond
// quote selection and backslashes is needed.
func pythonRepr(text string) string {
	quote := "'"
	if strings.Contains(text, "'") && !strings.Contains(text, `"`) {
		quote = `"`
	}
	escaped := strings.ReplaceAll(text, `\`, `\\`)
	if quote == "'" {
		escaped = strings.ReplaceAll(escaped, "'", `\'`)
	}
	return quote + escaped + quote
}
