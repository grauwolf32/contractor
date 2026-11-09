package public

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgconn"
)

type contractCostReader struct{ err error }

func (s *contractCostReader) GetLLMCost(_ context.Context, owner, scope, id string) (reporting.LLMCostSummary, error) {
	if s.err != nil {
		return reporting.LLMCostSummary{}, s.err
	}
	if owner == "" || id == "foreign" {
		return reporting.LLMCostSummary{}, telemetry.ErrNotFound
	}
	total := reporting.NewLLMCostTotals(false)
	contribution := reporting.CostContribution{ReportID: "cost-report", ReceivedAt: time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC), Complete: true, ModelCalls: 1, Cost: &reporting.LLMCost{Currency: "USD", Calls: 1, Groups: []reporting.LLMCostGroup{{Model: "model", Phase: "main", Source: "provider", Calls: 1, Amount: "0.00000625"}}}}
	if err := total.ReplaceCostContribution(nil, &contribution, "worker", scope != "run", contribution.ReceivedAt); err != nil {
		return reporting.LLMCostSummary{}, err
	}
	return reporting.BuildLLMCostSummary(scope, id, total, false, true, contribution.ReceivedAt), nil
}

func TestPublicLLMCostContracts(t *testing.T) {
	reader := &contractCostReader{}
	f := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(s *testServices) { s.Costs = reader })
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, plural := range []string{"runs", "audits", "projects"} {
		path := "/v1/" + plural + "/cost-item/llm-cost"
		response := serveAndValidatePublicContract(t, router, f.handler, newPublicContractRequest(http.MethodGet, path, nil), true)
		if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" || response.Header().Get("ETag") != "" {
			t.Fatalf("cost response %s: %d %v %s", plural, response.Code, response.Header(), response.Body.String())
		}
		for _, test := range []struct {
			suffix string
			status int
		}{{"foreign/llm-cost", http.StatusNotFound}, {"cost-item/llm-cost?unexpected=1", http.StatusBadRequest}} {
			r := serveAndValidatePublicContract(t, router, f.handler, newPublicContractRequest(http.MethodGet, "/v1/"+plural+"/"+test.suffix, nil), false)
			if r.Code != test.status {
				t.Fatalf("cost failure: %d %s", r.Code, r.Body.String())
			}
		}
	}
	for _, query := range []string{"executionId=one&executionId=two", "executionId=one&executionId=one"} {
		r := serveAndValidatePublicContract(t, router, f.handler, newPublicContractRequest(http.MethodGet, "/v1/audits/cost-item/llm-cost?"+query, nil), false)
		want := http.StatusOK
		if query == "executionId=one&executionId=one" {
			want = http.StatusBadRequest
		}
		if r.Code != want {
			t.Fatalf("execution selection: %d %s", r.Code, r.Body.String())
		}
	}
	reader.err = &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}
	r := serveAndValidatePublicContract(t, router, f.handler, newPublicContractRequest(http.MethodGet, "/v1/runs/cost-item/llm-cost", nil), true)
	if r.Code != http.StatusServiceUnavailable {
		t.Fatalf("retry classification: %d %s", r.Code, r.Body.String())
	}
}

func (s *contractCostReader) GetAuditExecutionCosts(ctx context.Context, owner, auditID string, ids []string) ([]reporting.LLMCostExecution, error) {
	if _, err := s.GetLLMCost(ctx, owner, "audit", auditID); err != nil {
		return nil, err
	}
	items := []reporting.LLMCostExecution{}
	for _, id := range ids {
		items = append(items, reporting.LLMCostExecution{ExecutionID: id, Basis: "none", Coverage: "unavailable"})
	}
	return items, nil
}
