package public

import (
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type auditTaskCreationRequest struct {
	Assertions map[string]auditservice.HypothesisContent `json:"assertions,omitempty"`
	Attention  []string                                  `json:"attention,omitempty"`
	Work       []auditcontract.PreparedSelection         `json:"work"`
	Hypotheses map[string]string                         `json:"hypotheses,omitempty"`
}

type auditTaskResponse struct {
	Findings      *auditstore.CanonicalFindingResult `json:"findings,omitempty"`
	Hypothesis    *auditcontract.Hypothesis          `json:"hypothesis_snapshot,omitempty"`
	Attention     bool                               `json:"attention"`
	Document      auditcontract.Task                 `json:"document"`
	Source        auditstore.TaskSource              `json:"source"`
	Ordinal       int                                `json:"ordinal"`
	Iteration     int                                `json:"iteration"`
	State         auditstore.TaskState               `json:"state"`
	Attempts      int                                `json:"attempts"`
	LastExecution *string                            `json:"last_execution_id,omitempty"`
	Result        *auditcontract.Material            `json:"result,omitempty"`
	GapCode       *string                            `json:"gap_code,omitempty"`
	CreatedAt     time.Time                          `json:"created_at"`
	UpdatedAt     time.Time                          `json:"updated_at"`
}

func auditTaskReadModel(task auditstore.Task) auditTaskResponse {
	return auditTaskResponse{Findings: task.Findings, Hypothesis: task.Hypothesis, Attention: task.Attention, Document: task.Document, Source: task.Source, Ordinal: task.Ordinal,
		Iteration: task.Iteration, State: task.State, Attempts: task.Attempts, LastExecution: task.LastExecution,
		Result: task.Result, GapCode: task.GapCode, CreatedAt: task.CreatedAt, UpdatedAt: task.UpdatedAt}
}

// Task pages budget canonical metadata. Preserve that encoding at the HTTP
// boundary: HTML escaping could otherwise expand accepted text sixfold.
// The store reserves 48 KiB per row for result/progress and page framing.
func writeAuditTaskJSON(w http.ResponseWriter, value any) error {
	data, err := strictjson.Canonical(value)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, err = w.Write(data)
	return err
}

func (h *auditHandlers) createAuditTasks(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	key, revision, err := auditMutationHeaders(r)
	if err != nil {
		return err
	}
	var request auditTaskCreationRequest
	if err := decodeJSONBounded(w, r, &request, auditcontract.MaxMutationBytes); err != nil {
		return err
	}
	if len(request.Work) < 1 || len(request.Work) > auditcontract.MaxMutationRecords {
		return errInvalidRequest
	}
	auditID := r.PathValue("auditId")
	params := auditservice.TaskCreationParams{OwnerID: principalUserID(r.Context()), AuditID: auditID, ExpectedRevision: revision, IdempotencyKey: key, Work: request.Work, Hypotheses: request.Hypotheses, Attention: request.Attention, Assertions: request.Assertions}
	digest, err := auditservice.TaskCreationRequestDigest(params)
	if err != nil {
		return errInvalidRequest
	}
	params.RequestDigest = digest
	result, err := h.Audits.CreateTasks(r.Context(), params)
	if err != nil {
		return err
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Acceptance.AuditRevision, 10)))
	// The receipt describes the original mutation even if the Audit changed
	// after it committed. No mutable read participates in acknowledgement.
	return writeAuditTaskJSON(w, result.Acceptance)
}

func (h *auditHandlers) listAuditTasks(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encoded, err := pageQuery(r.URL.RawQuery, "state", "hypothesis_id")
	if err != nil {
		return err
	}
	params := auditstore.ListTasksParams{OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: limit}
	state := ""
	if values, exists := query["state"]; exists {
		state = values[0]
		candidate := auditstore.TaskState(state)
		if candidate != auditstore.TaskReady && candidate != auditstore.TaskSelected && candidate != auditstore.TaskSettled {
			return errInvalidRequest
		}
		params.State = &candidate
	}
	if values, exists := query["hypothesis_id"]; exists {
		params.HypothesisID = values[0]
		if params.HypothesisID == "" {
			return errInvalidRequest
		}
	}
	kind := "audit-tasks:" + params.AuditID + ":" + state + ":" + params.HypothesisID
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 1)
	if err != nil {
		return err
	}
	if len(cursor) != 0 {
		params.AfterOrdinal, err = strconv.Atoi(cursor[0])
		if err != nil || params.AfterOrdinal < 1 {
			return errInvalidRequest
		}
	}
	result, err := h.Audits.Tasks(r.Context(), params)
	if err != nil {
		return err
	}
	page := pageInfoResponse{}
	if result.NextOrdinal != nil {
		next, err := h.cursors.encodePageCursor(kind, strconv.Itoa(*result.NextOrdinal))
		if err != nil {
			return err
		}
		page.HasMore, page.NextCursor = true, &next
	}
	items := make([]auditTaskResponse, len(result.Tasks))
	for i, task := range result.Tasks {
		items[i] = auditTaskReadModel(task)
	}
	return writeAuditTaskJSON(w, struct {
		Items []auditTaskResponse `json:"items"`
		Page  pageInfoResponse    `json:"page"`
	}{Items: items, Page: page})
}

func (h *auditHandlers) getAuditTask(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	task, err := h.Audits.Task(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"), r.PathValue("taskId"))
	if err != nil {
		return err
	}
	return writeAuditTaskJSON(w, auditTaskReadModel(task))
}
