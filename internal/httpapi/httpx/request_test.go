package httpx

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

var errTestInvalid = errors.New("test invalid")

func TestExactQuery(t *testing.T) {
	for _, tc := range []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{name: "empty", raw: ""},
		{name: "allowed", raw: "limit=1&cursor=x"},
		{name: "unknown", raw: "other=1", wantErr: true},
		{name: "repeated", raw: "limit=1&limit=2", wantErr: true},
		{name: "malformed", raw: "limit=%zz", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ExactQuery(errTestInvalid, tc.raw, "limit", "cursor")
			if (err != nil) != tc.wantErr {
				t.Fatalf("ExactQuery(%q) error = %v, want error %v", tc.raw, err, tc.wantErr)
			}
			if err != nil && !errors.Is(err, errTestInvalid) {
				t.Fatalf("ExactQuery(%q) error = %v, want wrapped sentinel", tc.raw, err)
			}
		})
	}
}

func TestExactQueryWithRepeated(t *testing.T) {
	values, err := ExactQueryWithRepeated(errTestInvalid, "tag=a&tag=b&limit=1", "tag", 2, "limit")
	if err != nil || len(values["tag"]) != 2 {
		t.Fatalf("ExactQueryWithRepeated = %v, %v", values, err)
	}
	if _, err := ExactQueryWithRepeated(errTestInvalid, "tag=a&tag=b&tag=c", "tag", 2); !errors.Is(err, errTestInvalid) {
		t.Fatalf("over-bound repeated key error = %v", err)
	}
}

func TestRequestMediaType(t *testing.T) {
	for _, tc := range []struct {
		name    string
		values  []string
		want    string
		wantErr bool
	}{
		{name: "plain", values: []string{"application/json"}, want: "application/json"},
		{name: "missing", wantErr: true},
		{name: "repeated", values: []string{"text/plain", "text/plain"}, wantErr: true},
		{name: "parameters", values: []string{"text/plain; charset=utf-8"}, wantErr: true},
		// mime.ParseMediaType already folds case, so the lowercase guard accepts this.
		{name: "uppercase", values: []string{"Text/Plain"}, want: "text/plain"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPut, "/", nil)
			for _, value := range tc.values {
				r.Header.Add("Content-Type", value)
			}
			got, err := RequestMediaType(errTestInvalid, r)
			if (err != nil) != tc.wantErr || got != tc.want {
				t.Fatalf("RequestMediaType = %q, %v", got, err)
			}
			if err != nil && !errors.Is(err, errTestInvalid) {
				t.Fatalf("RequestMediaType error = %v, want wrapped sentinel", err)
			}
		})
	}
}

func TestParseStrongETag(t *testing.T) {
	for _, tc := range []struct {
		raw     string
		want    string
		wantErr error
	}{
		{raw: `"7"`, want: "7"},
		{raw: `  "rev-1" `, want: "rev-1"},
		{raw: `W/"7"`, wantErr: ErrWeakETag},
		{raw: `"7", "8"`, wantErr: ErrWeakETag},
		{raw: `7`, wantErr: ErrUnquotedETag},
		{raw: `""`, wantErr: ErrUnquotedETag},
	} {
		got, err := ParseStrongETag(tc.raw)
		if got != tc.want || !errors.Is(err, tc.wantErr) || (tc.wantErr == nil && err != nil) {
			t.Fatalf("ParseStrongETag(%q) = %q, %v; want %q, %v", tc.raw, got, err, tc.want, tc.wantErr)
		}
	}
}

func TestQuotedETag(t *testing.T) {
	if got := QuotedETag(nil); got != "" {
		t.Fatalf("QuotedETag(nil) = %q", got)
	}
	revision := `a"b`
	if got := QuotedETag(&revision); got != `"a\"b"` {
		t.Fatalf("QuotedETag = %q", got)
	}
}
