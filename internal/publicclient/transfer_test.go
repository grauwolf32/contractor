package publicclient

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const (
	transferTimeout  = 500 * time.Millisecond
	transferInterval = 50 * time.Millisecond
	transferChunks   = 20 // 1s in total: twice the timeout, a tenth of it per chunk.
)

func slowListServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(APIVersionHeader, APIVersion)
		w.Header().Set("Content-Type", "application/json")
		flusher := w.(http.Flusher)
		for range transferChunks {
			_, _ = w.Write([]byte(" "))
			flusher.Flush()
			select {
			case <-time.After(transferInterval):
			case <-r.Context().Done():
				return
			}
		}
		_, _ = w.Write([]byte(`{"items":[],"page":{"hasMore":false}}`))
	}))
	t.Cleanup(server.Close)
	return server
}

func TestTransferTimeoutBoundsInactivityNotDuration(t *testing.T) {
	server := slowListServer(t)
	whole, err := New(Options{Server: server.URL, Token: "secret", Timeout: transferTimeout})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := whole.API.ListWorkflowsWithResponse(t.Context(), nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("whole-request timeout = %v, want a deadline", err)
	}
	transfer, err := New(Options{Server: server.URL, Token: "secret", Timeout: transferTimeout, Transfer: true})
	if err != nil {
		t.Fatal(err)
	}
	response, err := transfer.API.ListWorkflowsWithResponse(t.Context(), nil)
	if err != nil || response.JSON200 == nil {
		t.Fatalf("steady slow download = %+v, %v", response, err)
	}
}

type slowReader struct{ remaining int }

func (r *slowReader) Read(p []byte) (int, error) {
	if r.remaining == 0 {
		return 0, io.EOF
	}
	time.Sleep(transferInterval)
	r.remaining--
	p[0] = 'x'
	return 1, nil
}

func TestTransferTimeoutAllowsSteadySlowUpload(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil || len(body) != transferChunks {
			t.Errorf("uploaded %d bytes: %v", len(body), err)
		}
		w.Header().Set(APIVersionHeader, APIVersion)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()
	transfer, err := New(Options{Server: server.URL, Token: "secret", Timeout: transferTimeout, Transfer: true})
	if err != nil {
		t.Fatal(err)
	}
	response, err := transfer.API.PutArtifactWithBodyWithResponse(t.Context(), "documents", "brief", nil, "text/plain", &slowReader{remaining: transferChunks})
	if err != nil || response.StatusCode() != http.StatusCreated {
		t.Fatalf("steady slow upload = %v, %v", response, err)
	}
}

func TestTransferTimeoutFailsAStalledExchange(t *testing.T) {
	for _, stage := range []string{"headers", "body"} {
		t.Run(stage, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if stage == "body" {
					w.Header().Set(APIVersionHeader, APIVersion)
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(" "))
					w.(http.Flusher).Flush()
				}
				<-r.Context().Done()
			}))
			defer server.Close()
			transfer, err := New(Options{Server: server.URL, Token: "secret", Timeout: transferTimeout, Transfer: true})
			if err != nil {
				t.Fatal(err)
			}
			started := time.Now()
			_, err = transfer.API.ListWorkflowsWithResponse(t.Context(), nil)
			if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "no progress") {
				t.Fatalf("stalled %s = %v", stage, err)
			}
			if elapsed := time.Since(started); elapsed > 10*transferTimeout {
				t.Fatalf("stalled %s took %s", stage, elapsed)
			}
		})
	}
}
