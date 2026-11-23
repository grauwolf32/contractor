package public

import (
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAuditTimeLimitRequest(t *testing.T) {
	for _, test := range []struct {
		body    string
		valid   bool
		seconds *int
	}{
		{"", true, nil}, {"{}", true, nil}, {`{"deadlineSeconds":0}`, true, timeLimitPointer(0)},
		{`{"deadlineSeconds":604800}`, true, timeLimitPointer(604800)},
		{`{"deadlineSeconds":-1}`, false, nil}, {`{"deadlineSeconds":31536001}`, false, nil},
		{`{"deadlineSeconds":null}`, false, nil}, {`{"deadlineSeconds":1.5}`, false, nil},
		{`{"unknown":1}`, false, nil}, {`null`, false, nil}, {`{} {}`, false, nil},
	} {
		t.Run(test.body, func(t *testing.T) {
			request := httptest.NewRequest("POST", "/", strings.NewReader(test.body))
			request.Header.Set("Content-Type", "application/json")
			got, err := readAuditTimeLimit(httptest.NewRecorder(), request)
			if !test.valid {
				if !errors.Is(err, errInvalidRequest) {
					t.Fatalf("error=%v", err)
				}
				return
			}
			if err != nil || (got == nil) != (test.seconds == nil) || got != nil && *got != *test.seconds {
				t.Fatalf("limit=%v error=%v", got, err)
			}
		})
	}
	oversized := httptest.NewRequest("POST", "/", strings.NewReader(`{"deadlineSeconds":1`+strings.Repeat(" ", 1024)+`}`))
	oversized.Header.Set("Content-Type", "application/json")
	if _, err := readAuditTimeLimit(httptest.NewRecorder(), oversized); !errors.Is(err, errRequestTooLarge) {
		t.Fatalf("oversized time limit error=%v", err)
	}
}

func timeLimitPointer(value int) *int { return &value }
