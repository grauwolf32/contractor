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
	Work []auditcontract.PreparedSelection `json:"work"`
}

type auditTaskResponse struct {
	Document      auditcontract.Task      `json:"document"`
	Source        auditstore.TaskSource   `json:"source"`
	Ordinal       int                     `json:"ordinal"`
	Iteration     int                     `json:"iteration"`
	State         auditstore.TaskState    `json:"state"`
	Attempts      int                     `json:"attempts"`
	LastExecution *string                 `json:"last_execution_id,omitempty"`
	Result        *auditcontract.Material `json:"result,omitempty"`
	GapCode       *string                 `json:"gap_code,omitempty"`
	CreatedAt     time.Time               `json:"created_at"`
	UpdatedAt     time.Time               `json:"updated_at"`
}

func auditTaskReadModel(task auditstore.Task) auditTaskResponse {
	return auditTaskResponse{Document: task.Document, Source: task.Source, Ordinal: task.Ordinal,
		Iteration: task.Iteration, State: task.State, Attempts: task.Attempts, LastExecution: task.LastExecution,
		Result: task.Result, GapCode: task.GapCode, CreatedAt: task.CreatedAt, UpdatedAt: task.UpdatedAt}
}

// Task pages budget canonical metadata. Preserve that encoding at the HTTP
// boundary: HTML escaping could otherwise expand accepted text sixfold.
// The store reserves 8 KiB per row for result/progress and page framing.
func writeAuditTaskJSON(w http.ResponseWriter, value any) {
	data, err := strictjson.Canonical(value)
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (h *auditHandlers) createAuditTasks(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if err := requireJSONMediaType(r); err != nil {
		handleError(w, err)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	key, revision, err := auditMutationHeaders(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var request auditTaskCreationRequest
	if err := decodeJSONBounded(w, r, &request, auditcontract.MaxMutationBytes); err != nil {
		handleError(w, err)
		return
	}
	if len(request.Work) < 1 || len(request.Work) > auditcontract.MaxMutationRecords {
		handleError(w, errInvalidRequest)
		return
	}
	auditID := r.PathValue("auditId")
	digest, err := auditservice.TaskCreationRequestDigest(auditID, revision, request.Work)
	if err != nil {
		handleError(w, errInvalidRequest)
		return
	}
	result, err := h.Audits.CreateTasks(r.Context(), auditservice.TaskCreationParams{
		OwnerID: principalUserID(r.Context()), AuditID: auditID, ExpectedRevision: revision,
		IdempotencyKey: key, RequestDigest: digest, Work: request.Work,
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Acceptance.AuditRevision, 10)))
	// The receipt describes the original mutation even if the Audit changed
	// after it committed. No mutable read participates in acknowledgement.
	writeAuditTaskJSON(w, result.Acceptance)
}

func (h *auditHandlers) listAuditTasks(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encoded, err := pageQuery(r.URL.RawQuery, "state")
	if err != nil {
		handleError(w, err)
		return
	}
	params := auditstore.ListTasksParams{OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: limit}
	state := ""
	if values, exists := query["state"]; exists {
		state = values[0]
		candidate := auditstore.TaskState(state)
		if candidate != auditstore.TaskReady && candidate != auditstore.TaskSelected && candidate != auditstore.TaskSettled {
			handleError(w, errInvalidRequest)
			return
		}
		params.State = &candidate
	}
	kind := "audit-tasks:" + params.AuditID + ":" + state
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 1)
	if err != nil {
		handleError(w, err)
		return
	}
	if len(cursor) != 0 {
		params.AfterOrdinal, err = strconv.Atoi(cursor[0])
		if err != nil || params.AfterOrdinal < 1 {
			handleError(w, errInvalidRequest)
			return
		}
	}
	result, err := h.Audits.Tasks(r.Context(), params)
	if err != nil {
		handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if result.NextOrdinal != nil {
		next, err := h.cursors.encodePageCursor(kind, strconv.Itoa(*result.NextOrdinal))
		if err != nil {
			handleError(w, err)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	items := make([]auditTaskResponse, len(result.Tasks))
	for i, task := range result.Tasks {
		items[i] = auditTaskReadModel(task)
	}
	writeAuditTaskJSON(w, struct {
		Items []auditTaskResponse `json:"items"`
		Page  pageInfoResponse    `json:"page"`
	}{Items: items, Page: page})
}

func (h *auditHandlers) getAuditTask(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	task, err := h.Audits.Task(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"), r.PathValue("taskId"))
	if err != nil {
		handleError(w, err)
		return
	}
	writeAuditTaskJSON(w, auditTaskReadModel(task))
}
