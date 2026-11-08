package public

import (
	"context"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/apierror"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

type LLMCostReader interface {
	GetLLMCost(context.Context, string, string, string) (reporting.LLMCostSummary, error)
	GetAuditExecutionCosts(context.Context, string, string, []string) ([]reporting.LLMCostExecution, error)
}

type CostDependencies struct{ Costs LLMCostReader }
type costHandlers struct{ CostDependencies }

func (h *costHandlers) getRunCost(w http.ResponseWriter, r *http.Request) error {
	return h.get(w, r, "run", r.PathValue("runId"))
}
func (h *costHandlers) getAuditCost(w http.ResponseWriter, r *http.Request) error {
	return h.get(w, r, "audit", r.PathValue("auditId"))
}
func (h *costHandlers) getProjectCost(w http.ResponseWriter, r *http.Request) error {
	return h.get(w, r, "project", r.PathValue("projectId"))
}

// Cost has independent operational freshness. A read never changes a parent
// revision or repeats inference; PostgreSQL transient conflicts retain the
// common storage_transaction_conflict/503 classification.
func (h *costHandlers) get(w http.ResponseWriter, r *http.Request, scope, id string) error {
	w.Header().Set("Cache-Control", "no-store")
	allowed := []string{}
	if scope == "audit" {
		allowed = append(allowed, "executionId")
	}
	query, err := exactQueryWithRepeated(r.URL.RawQuery, "executionId", 64, allowed...)
	if err != nil {
		return err
	}
	ids := query["executionId"]
	if len(ids) > 64 {
		return errInvalidRequest
	}
	seen := map[string]bool{}
	for _, executionID := range ids {
		if !contracts.ValidResourceID(executionID) || seen[executionID] {
			return errInvalidRequest
		}
		seen[executionID] = true
	}
	if !contracts.ValidResourceID(id) {
		return errInvalidRequest
	}
	value, err := h.Costs.GetLLMCost(r.Context(), principalUserID(r.Context()), scope, id)
	if err == nil && len(ids) > 0 {
		value.Executions, err = h.Costs.GetAuditExecutionCosts(r.Context(), principalUserID(r.Context()), id, ids)
	}
	if errors.Is(err, telemetry.ErrNotFound) {
		writeError(w, apierror.NotFound, "")
		return nil
	}
	if errors.Is(err, telemetry.ErrInvalid) {
		return errInvalidRequest
	}
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, value)
	return nil
}
