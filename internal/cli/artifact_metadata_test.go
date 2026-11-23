package cli

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestSharedArtifactStorageTimestampsCLI(t *testing.T) {
	var cases []struct {
		Name  string
		Valid bool
		Value json.RawMessage
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "artifact-storage-timestamps.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, scope := range []artifactScope{{}, {projectID: "project"}, {runID: "run"}} {
		for _, tc := range cases {
			t.Run(scope.projectID+scope.runID+"/"+tc.Name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(tc.Value)
				}))
				defer server.Close()
				client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "fixture-token", AllowHTTP: true, Timeout: time.Second})
				if err != nil {
					t.Fatal(err)
				}
				_, err = artifactMetadata(context.Background(), client, scope, "docs", "source", "r1")
				if (err == nil) != tc.Valid {
					t.Fatalf("valid=%t: %v", tc.Valid, err)
				}
				if tc.Valid {
					if _, err := artifactMetadata(context.Background(), client, scope, "docs", "source", "another"); err == nil {
						t.Fatal("accepted wrong exact revision")
					}
				}
			})
		}
	}
}
