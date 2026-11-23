package cli

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/publicclient"
)

// TestSharedIdempotencyKeyCasesBindExplicitCLIKeys sends project create with
// each shared --idempotency-key: a key outside the public grammar is a usage
// error that sends nothing, and a valid key reaches the Server unchanged. An
// omitted key is generated in the same grammar.
func TestSharedIdempotencyKeyCasesBindExplicitCLIKeys(t *testing.T) {
	var received []string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		received = append(received, request.Header.Values("Idempotency-Key")...)
		writer.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(writer, `{"projectId":"project-1","kind":"project","name":"demo","lifecycle":"active",`+
			`"revision":"1","activeChecks":"automatic","createdAt":"2026-10-23T00:00:00Z","updatedAt":"2026-10-23T00:00:00Z"}`)
	}))
	defer server.Close()
	create := func(arguments ...string) error {
		var stdout, stderr bytes.Buffer
		command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
			if name == "CONTRACTOR_API_TOKEN" {
				return "secret"
			}
			return ""
		})
		return command.Run(context.Background(), append([]string{"--server", server.URL, "project", "create", "demo"}, arguments...))
	}

	cases := contractstest.ReadIdempotencyKeyCases(t)
	for _, key := range cases.Valid {
		received = nil
		if err := create("--idempotency-key", key); err != nil {
			t.Fatalf("valid key %q: %v", key, err)
		}
		if len(received) != 1 || received[0] != key {
			t.Fatalf("valid key %q sent as %q", key, received)
		}
	}
	for _, key := range cases.Invalid {
		if key == "" {
			continue // An empty flag is an omitted key, generated below.
		}
		received = nil
		var usage *UsageError
		if err := create("--idempotency-key", key); !errors.As(err, &usage) || len(received) != 0 {
			t.Fatalf("invalid key %q: error %v, sent %q", key, err, received)
		}
	}
	received = nil
	if err := create(); err != nil || len(received) != 1 || !contracts.ValidIdempotencyKey(received[0]) {
		t.Fatalf("generated key: error %v, sent %q", err, received)
	}
}
