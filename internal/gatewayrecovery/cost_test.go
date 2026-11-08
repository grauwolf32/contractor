package gatewayrecovery

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

func TestSharedLLMCostObservations(t *testing.T) {
	var cases struct {
		Normalization []struct {
			Raw    string
			Amount *string
		}
		Observations []struct {
			Name, Model, Body string
			Headers           map[string]string
			Source, Amount    *string
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "llm-cost-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Normalization) == 0 || len(cases.Observations) == 0 {
		t.Fatal("empty cost fixtures")
	}
	for _, c := range cases.Normalization {
		t.Run(c.Raw, func(t *testing.T) {
			got, ok := reporting.NormalizeUSD(c.Raw)
			if ok != (c.Amount != nil) || ok && got != *c.Amount {
				t.Fatalf("NormalizeUSD=%q,%v", got, ok)
			}
		})
	}
	for _, c := range cases.Observations {
		t.Run(c.Name, func(t *testing.T) {
			headers := http.Header{}
			for k, v := range c.Headers {
				headers.Set(k, v)
			}
			source, amount := responseCost(c.Model, headers, []byte(c.Body))
			if c.Amount == nil {
				if source != "" || amount != "" {
					t.Fatal("unavailable became priced")
				}
			} else if source != *c.Source || amount != *c.Amount {
				t.Fatalf("cost=%q/%q", source, amount)
			}
		})
	}
}

func TestLLMCostPhysicalAttemptsKeepFailedAndRejectedResponses(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("llm_provider-x-litellm-response-cost", "0.1")
		if calls == 1 {
			w.WriteHeader(503)
		}
		_, _ = w.Write([]byte(`invalid model response`))
	}))
	defer server.Close()
	var cost reporting.LLMCostCollector
	ctx := WithLLMCostModel(WithLLMCostCollector(context.Background(), &cost), "selected-model")
	for i := 0; i < 2; i++ {
		phase := ctx
		if i == 1 {
			phase = WithLLMCostPhase(ctx, "compaction")
		}
		request, err := http.NewRequestWithContext(phase, http.MethodPost, server.URL, bytes.NewReader([]byte(`{}`)))
		if err != nil {
			t.Fatal(err)
		}
		_, err = Send(request, server.Client(), 1024, llmgateway.DefaultGatewayFailureSignatures())
		if (i == 0) != (err != nil) {
			t.Fatalf("inference outcome changed: %v", err)
		}
	}
	got := cost.Snapshot(0)
	if got.Calls != 2 || len(got.Groups) != 2 {
		t.Fatalf("cost=%+v", got)
	}
	for _, g := range got.Groups {
		if g.Source != "provider" || g.Amount != "0.1" || g.Calls != 1 {
			t.Fatalf("group=%+v", g)
		}
	}
}
