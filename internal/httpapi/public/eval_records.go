package public

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func (h *handler) ingestEvalResult(w http.ResponseWriter, r *http.Request) {
	h.ingestEvalRecord(w, r, "ResultInput", "result")
}
func (h *handler) assessEvalMember(w http.ResponseWriter, r *http.Request) {
	h.ingestEvalRecord(w, r, "AssessmentSubmission", "assessment")
}

func (h *handler) ingestEvalRecord(w http.ResponseWriter, r *http.Request, kind, operation string) {
	doc, mutation, ok := h.evalBody(w, r, kind, false)
	if !ok {
		return
	}
	if kind == "AssessmentSubmission" {
		var discriminator struct {
			SchemaVersion string `json:"schemaVersion"`
		}
		if err := json.Unmarshal(doc.Bytes(), &discriminator); err != nil {
			h.evalError(w, err)
			return
		}
		kind = "AssessmentInput"
		if discriminator.SchemaVersion == "contractor.eval-check-request/v1" {
			kind = "CheckRequest"
		}
		var err error
		doc, err = evaldomain.Freeze(kind, doc.Bytes())
		if err != nil {
			h.evalError(w, err)
			return
		}
	}
	id, member := r.PathValue("id"), r.PathValue("memberId")
	scope, err := h.dependencies.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), id+":"+member, operation, mutation.Key)
	if err != nil {
		h.evalError(w, err)
		return
	}
	receipt, err := h.dependencies.Evals.PutRecord(r.Context(), scope, id, member, doc, mutation)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalRecordReceipt(w, "RecordReceipt", receipt)
}

func (h *handler) evalRecordReceipt(w http.ResponseWriter, kind string, receipt evalstore.Receipt) {
	if receipt.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	h.evalJSON(w, http.StatusCreated, kind, json.RawMessage(receipt.Response))
	if h.dependencies.EvalNotifier != nil {
		h.dependencies.EvalNotifier.Wake()
	}
}

func (h *handler) selectEvalRecords(w http.ResponseWriter, r *http.Request) {
	doc, mutation, ok := h.evalBody(w, r, "SelectionInput", true)
	if !ok {
		return
	}
	var input evaldomain.SelectionInput
	if err := evaldomain.DecodeInto(doc.Kind(), doc.Bytes(), &input); err != nil {
		h.evalError(w, err)
		return
	}
	id := r.PathValue("id")
	scope, err := h.dependencies.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), id, "selection", mutation.Key)
	if err != nil {
		h.evalError(w, err)
		return
	}
	receipt, err := h.dependencies.Evals.Select(r.Context(), scope, id, input, mutation)
	if err != nil {
		h.evalError(w, err)
		return
	}
	var data struct {
		Revision int64 `json:"revision"`
	}
	if err = json.Unmarshal(receipt.Response, &data); err != nil {
		h.evalError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(data.Revision, 10)))
	h.evalRecordReceipt(w, "SelectionReceipt", receipt)
}

func (h *handler) reviewEvalMember(w http.ResponseWriter, r *http.Request) {
	q, err := exactQuery(r.URL.RawQuery, "resultSha256")
	if err != nil {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	if q.Has("resultSha256") && evaldomain.Validate("Digest", mustEvalJSON(q.Get("resultSha256"))) != nil {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	data, err := h.dependencies.Evals.Review(r.Context(), principalUserID(r.Context()), r.PathValue("id"), r.PathValue("memberId"), q.Get("resultSha256"))
	if err != nil {
		h.evalError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(data.Revision, 10)))
	h.evalJSON(w, http.StatusOK, "Review", data)
}

func mustEvalJSON(value string) []byte { raw, _ := json.Marshal(value); return raw }

func (h *handler) evalMemberExecutions(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r)
	if err != nil {
		h.evalError(w, err)
		return
	}
	cursor, err := h.readEvalCursor(r, q, 1)
	if err != nil {
		h.evalError(w, err)
		return
	}
	revision, err := cursorRevision(cursor)
	if err != nil {
		h.evalError(w, err)
		return
	}
	after := ""
	if len(cursor.Position) > 0 {
		after = cursor.Position[0]
	}
	data, err := h.dependencies.Evals.Executions(r.Context(), principalUserID(r.Context()), r.PathValue("id"), r.PathValue("memberId"), after, limit, revision)
	if err != nil {
		h.evalError(w, err)
		return
	}
	page, err := h.evalPage(r, q, data.HasMore, strconv.FormatInt(data.Revision, 10), data.LastKey)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, http.StatusOK, "ExecutionPage", map[string]any{"inventoryRevision": data.Revision, "inventoryComplete": data.Complete, "items": data.Items, "gaps": data.Gaps, "page": page})
}
