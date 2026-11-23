package public

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func (h *evalHandlers) ingestEvalResult(w http.ResponseWriter, r *http.Request) error {
	return h.ingestEvalRecord(w, r, "ResultInput", "result")
}
func (h *evalHandlers) assessEvalMember(w http.ResponseWriter, r *http.Request) error {
	return h.ingestEvalRecord(w, r, "AssessmentSubmission", "assessment")
}

func (h *evalHandlers) ingestEvalRecord(w http.ResponseWriter, r *http.Request, kind, operation string) error {
	doc, mutation, ok := evalBody(w, r, kind, false)
	if !ok {
		return nil
	}
	if kind == "AssessmentSubmission" {
		var discriminator struct {
			SchemaVersion string `json:"schemaVersion"`
		}
		if err := json.Unmarshal(doc.Bytes(), &discriminator); err != nil {
			return err
		}
		kind = "AssessmentInput"
		if discriminator.SchemaVersion == "contractor.eval-check-request/v1" {
			kind = "CheckRequest"
		}
		var err error
		doc, err = evaldomain.Freeze(kind, doc.Bytes())
		if err != nil {
			return err
		}
	}
	id, member := r.PathValue("id"), r.PathValue("memberId")
	scope, err := h.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), id+":"+member, operation, mutation.Key)
	if err != nil {
		return err
	}
	receipt, err := h.Evals.PutRecord(r.Context(), scope, id, member, doc, mutation)
	if err != nil {
		return err
	}
	h.evalRecordReceipt(w, "RecordReceipt", receipt)
	return nil
}

func (h *evalHandlers) evalRecordReceipt(w http.ResponseWriter, kind string, receipt evalstore.Receipt) {
	if receipt.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	evalJSON(w, http.StatusCreated, kind, json.RawMessage(receipt.Response))
	h.EvalNotifier.Wake()
}

func (h *evalHandlers) selectEvalRecords(w http.ResponseWriter, r *http.Request) error {
	doc, mutation, ok := evalBody(w, r, "SelectionInput", true)
	if !ok {
		return nil
	}
	var input evaldomain.SelectionInput
	if err := evaldomain.DecodeInto(doc.Kind(), doc.Bytes(), &input); err != nil {
		return err
	}
	id := r.PathValue("id")
	scope, err := h.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), id, "selection", mutation.Key)
	if err != nil {
		return err
	}
	receipt, err := h.Evals.Select(r.Context(), scope, id, input, mutation)
	if err != nil {
		return err
	}
	var data struct {
		Revision int64 `json:"revision"`
	}
	if err = json.Unmarshal(receipt.Response, &data); err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(data.Revision, 10)))
	h.evalRecordReceipt(w, "SelectionReceipt", receipt)
	return nil
}

func (h *evalHandlers) reviewEvalMember(w http.ResponseWriter, r *http.Request) error {
	q, err := exactQuery(r.URL.RawQuery, "resultSha256")
	if err != nil {
		return evaldomain.Failure("eval_invalid")
	}
	if q.Has("resultSha256") && evaldomain.Validate("Digest", mustEvalJSON(q.Get("resultSha256"))) != nil {
		return evaldomain.Failure("eval_invalid")
	}
	data, err := h.Evals.Review(r.Context(), principalUserID(r.Context()), r.PathValue("id"), r.PathValue("memberId"), q.Get("resultSha256"))
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(data.Revision, 10)))
	evalJSON(w, http.StatusOK, "Review", data)
	return nil
}

func mustEvalJSON(value string) []byte { raw, _ := json.Marshal(value); return raw }

func (h *evalHandlers) listEvalMemberExecutions(w http.ResponseWriter, r *http.Request) error {
	q, limit, err := evalQuery(r)
	if err != nil {
		return err
	}
	cursor, err := h.readEvalCursor(r, q, 1)
	if err != nil {
		return err
	}
	revision, err := cursorRevision(cursor)
	if err != nil {
		return err
	}
	after := ""
	if len(cursor.Position) > 0 {
		after = cursor.Position[0]
	}
	data, err := h.Evals.Executions(r.Context(), principalUserID(r.Context()), r.PathValue("id"), r.PathValue("memberId"), after, limit, revision)
	if err != nil {
		return err
	}
	page, err := h.evalPage(r, q, data.HasMore, strconv.FormatInt(data.Revision, 10), data.LastKey)
	if err != nil {
		return err
	}
	evalJSON(w, http.StatusOK, "ExecutionPage", map[string]any{"inventoryRevision": data.Revision, "inventoryComplete": data.Complete, "items": data.Items, "gaps": data.Gaps, "page": page})
	return nil
}
