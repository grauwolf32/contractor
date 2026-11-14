package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
)

func TestOpenAPIValidatorRejectsInvalidArtifactsWithoutExternalReads(t *testing.T) {
	data, err := os.ReadFile("../private/recorded-results.json")
	if err != nil {
		t.Fatal(err)
	}
	var samples map[string]struct {
		OpenAPI json.RawMessage `json:"openapi"`
	}
	if err := json.Unmarshal(data, &samples); err != nil {
		t.Fatal(err)
	}
	valid := string(samples["api-dependency"].OpenAPI)
	if err := validate([]byte(valid)); err != nil {
		t.Fatal(err)
	}
	var reads atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reads.Add(1)
		_, _ = w.Write([]byte(`{"type":"string"}`))
	}))
	defer server.Close()
	for name, value := range map[string]string{
		"null":         "null",
		"trailing":     valid + " {}",
		"duplicate":    strings.Replace(valid, `"openapi": "3.1.0"`, `"openapi": "3.1.0", "openapi": "3.1.0"`, 1),
		"oversize":     strings.Repeat(" ", maxDocument+1),
		"missing name": strings.Replace(valid, `"name": "X-Tenant",`, "", 1),
		"missing in":   strings.Replace(valid, `"in": "header",`, "", 1),
		"remote ref":   `{"openapi":"3.1.0","info":{"title":"test","version":"1"},"paths":{"/test":{"$ref":"` + server.URL + `"}}}`,
		"missing ref":  `{"openapi":"3.1.0","info":{"title":"test","version":"1"},"paths":{"/test":{"$ref":"#/components/pathItems/missing"}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := validate([]byte(value)); err == nil {
				t.Fatal("invalid OpenAPI accepted")
			}
		})
	}
	if reads.Load() != 0 {
		t.Fatal("validator followed an external reference")
	}
}
