package public

import (
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func (h *auditHandlers) listAuditMaterials(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encoded, err := pageQuery(r.URL.RawQuery, "auditRevision")
	if err != nil {
		return err
	}
	id := r.PathValue("auditId")
	kind := "audit-materials:" + id
	cursor, revision, err := h.decodeAuditPageCursor(encoded, kind, query.Get("auditRevision"), 2)
	if err != nil {
		return err
	}
	if revision != nil && *revision > contracts.MaxExactInteger {
		return errInvalidRequest
	}
	after := int64(0)
	if len(cursor) > 0 {
		after, err = strconv.ParseInt(cursor[0], 10, 64)
		if err != nil || after < 1 {
			return errInvalidRequest
		}
	}
	result, err := h.Audits.Materials(r.Context(), auditstore.MaterialListParams{OwnerID: principalUserID(r.Context()), AuditID: id, AfterOrdinal: after, Limit: limit, AuditRevision: revision})
	if err != nil {
		return err
	}
	page := pageInfoResponse{}
	if result.NextOrdinal != nil {
		next, err := h.cursors.encodePageCursor(kind, strconv.FormatInt(*result.NextOrdinal, 10), strconv.FormatUint(result.AuditRevision, 10))
		if err != nil {
			return err
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	return writeAuditTaskJSON(w, struct {
		Items         []auditstore.MaterialEntry `json:"items"`
		Page          pageInfoResponse           `json:"page"`
		AuditRevision uint64                     `json:"auditRevision"`
	}{result.Items, page, result.AuditRevision})
}

func (h *auditHandlers) listAuditTaskAttempts(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encoded, err := pageQuery(r.URL.RawQuery, "auditRevision")
	if err != nil {
		return err
	}
	id, taskID := r.PathValue("auditId"), r.PathValue("taskId")
	kind := "audit-task-attempts:" + id + ":" + taskID
	cursor, revision, err := h.decodeAuditPageCursor(encoded, kind, query.Get("auditRevision"), 2)
	if err != nil {
		return err
	}
	if revision != nil && *revision > contracts.MaxExactInteger {
		return errInvalidRequest
	}
	after := 0
	if len(cursor) > 0 {
		after, err = strconv.Atoi(cursor[0])
		if err != nil || after < 1 {
			return errInvalidRequest
		}
	}
	result, err := h.Audits.TaskAttempts(r.Context(), auditstore.TaskAttemptListParams{OwnerID: principalUserID(r.Context()), AuditID: id, TaskID: taskID, AfterAttempt: after, Limit: limit, AuditRevision: revision})
	if err != nil {
		return err
	}
	page := pageInfoResponse{}
	if result.NextAttempt != nil {
		next, err := h.cursors.encodePageCursor(kind, strconv.Itoa(*result.NextAttempt), strconv.FormatUint(result.AuditRevision, 10))
		if err != nil {
			return err
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	return writeAuditTaskJSON(w, struct {
		Items         []auditstore.TaskAttempt `json:"items"`
		Page          pageInfoResponse         `json:"page"`
		AuditRevision uint64                   `json:"auditRevision"`
	}{result.Items, page, result.AuditRevision})
}
