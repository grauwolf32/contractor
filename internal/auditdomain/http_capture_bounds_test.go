package auditdomain

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestCapturedHTTPExchangeReadsItsDeclaredBodyBounds(t *testing.T) {
	for _, tc := range []struct {
		name              string
		request, response int
		valid             bool
	}{
		{"beyond ordinary document string limit", 65537, 65537, true},
		{"maximum request", MaximumHTTPRequestBodyBytes, 0, true},
		{"maximum response", 0, MaximumHTTPResponseBodyBytes, true},
		{"oversized request", MaximumHTTPRequestBodyBytes + 1, 0, false},
		{"oversized response", 0, MaximumHTTPResponseBodyBytes + 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headers := []FindingHTTPHeader{}
			requestBody := base64.StdEncoding.EncodeToString([]byte(strings.Repeat("x", tc.request)))
			responseBody := base64.StdEncoding.EncodeToString([]byte(strings.Repeat("x", tc.response)))
			value := CapturedHTTPExchange{Source: "http", RequestID: "1", RequestTag: "tag", Attempts: []CapturedHTTPAttempt{{Request: &CapturedHTTPRequest{Method: "POST", URL: "https://example.test/", Message: HTTPMessage{Kind: "structured", Headers: &headers, BodyBase64: &requestBody}}, Response: &CapturedHTTPResponse{Status: 200, Complete: true, Message: HTTPMessage{Kind: "structured", Headers: &headers, BodyBase64: &responseBody}}}}}
			data, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			var decoded CapturedHTTPExchange
			if err := strictjson.Decode(data, &decoded); (err == nil) != tc.valid {
				t.Fatalf("valid=%v bytes=%d: %v", tc.valid, len(data), err)
			}
		})
	}
}
