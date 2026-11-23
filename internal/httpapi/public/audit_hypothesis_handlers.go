package public

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func (h *auditHandlers) writeAuditHypothesis(w http.ResponseWriter, r *http.Request) error {
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
	var content auditservice.HypothesisContent
	if err := decodeJSONBounded(w, r, &content, 32768); err != nil {
		return err
	}
	if content.Key == "" || strings.TrimSpace(content.Statement) == "" || strings.TrimSpace(content.Grounds) == "" || content.Scope == nil {
		return errInvalidRequest
	}
	id, auditID := r.PathValue("hypothesisId"), r.PathValue("auditId")
	digest, err := auditservice.HypothesisRequestDigest(auditID, id, revision, content)
	if err != nil {
		return errInvalidRequest
	}
	result, replayed, err := h.Audits.WriteHypothesis(r.Context(), auditservice.HypothesisMutationParams{MutationParams: auditservice.MutationParams{OwnerID: principalUserID(r.Context()), AuditID: auditID, ExpectedRevision: revision, IdempotencyKey: key, RequestDigest: digest}, HypothesisID: id, Content: content})
	if err != nil {
		return err
	}
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.AuditRevision, 10)))
	return writeAuditTaskJSON(w, result)
}

func (h *auditHandlers) listAuditHypotheses(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	_, limit, encoded, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	params := auditstore.ListHypothesesParams{OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: limit}
	kind := "audit-hypotheses:" + params.AuditID
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 1)
	if err != nil {
		return err
	}
	if len(cursor) > 0 {
		params.AfterOrdinal, err = strconv.Atoi(cursor[0])
		if err != nil || params.AfterOrdinal < 1 {
			return errInvalidRequest
		}
	}
	result, err := h.Audits.Hypotheses(r.Context(), params)
	if err != nil {
		return err
	}
	page := pageInfoResponse{}
	if result.NextOrdinal != nil {
		next, err := h.cursors.encodePageCursor(kind, strconv.Itoa(*result.NextOrdinal))
		if err != nil {
			return err
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	return writeAuditTaskJSON(w, struct {
		CheckingWorkflows []string                `json:"checking_workflows"`
		PendingFixedCount int                     `json:"pending_fixed_count"`
		FixedDenominator  int                     `json:"fixed_denominator"`
		Items             []auditstore.Hypothesis `json:"items"`
		Page              pageInfoResponse        `json:"page"`
	}{append([]string{}, result.CheckingWorkflows...), result.PendingFixedCount, result.FixedDenominator, result.Items, page})
}

func (h *auditHandlers) getAuditHypothesis(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		return err
	}
	value, ok := query["revision"]
	if !ok || len(value) != 1 {
		return errInvalidRequest
	}
	revision, err := strconv.ParseUint(value[0], 10, 64)
	if err != nil || strconv.FormatUint(revision, 10) != value[0] || revision == 0 {
		return errInvalidRequest
	}
	result, err := h.Audits.Hypothesis(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"), r.PathValue("hypothesisId"), revision)
	if err != nil {
		return err
	}
	return writeAuditTaskJSON(w, result)
}
