package public

import (
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
)

type findingPageResponse struct {
	auditstore.FindingPageBasis
	Items []auditstore.ProjectFindingSummary `json:"items"`
	Page  pageInfoResponse                   `json:"page"`
}

func (h *auditHandlers) listAuditFindings(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursorValue, err := pageQuery(r.URL.RawQuery, "state", "verdict", "severity", "auditRevision", "projectRevision")
	if err != nil {
		return err
	}
	params := auditstore.FindingListParams{
		OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: pagelimit.FetchRows(limit),
	}
	stateValue, verdictValue, severityValue := "", "", ""
	if values, ok := query["state"]; ok {
		value := auditstore.FindingState(values[0])
		if !value.Valid() {
			return errInvalidRequest
		}
		params.State, stateValue = &value, values[0]
	}
	if values, ok := query["verdict"]; ok {
		verdictValue = values[0]
		if values[0] == "unreviewed" {
			params.Unreviewed = true
		} else {
			value := auditstore.AnalystVerdict(values[0])
			if value != auditstore.VerdictTruePositive && value != auditstore.VerdictFalsePositive {
				return errInvalidRequest
			}
			params.Verdict = &value
		}
	}
	if values, ok := query["severity"]; ok {
		value := auditstore.FindingSeverity(values[0])
		if !value.Valid() {
			return errInvalidRequest
		}
		params.Severity, severityValue = &value, values[0]
	}
	cursorKind := "audit-findings:" + params.OwnerID + ":" + params.AuditID + ":" + stateValue + ":" + verdictValue + ":" + severityValue
	cursor, revision, err := h.decodeAuditPageCursor(cursorValue, cursorKind, query.Get("auditRevision"), 4)
	if err != nil {
		return err
	}
	projectRevision, err := optionalProjectRevision(query.Get("projectRevision"))
	if err != nil {
		return err
	}
	if len(cursor) != 0 {
		parsed, parseErr := optionalProjectRevision(cursor[2])
		if parseErr != nil || parsed == nil || (projectRevision != nil && *projectRevision != *parsed) {
			return errInvalidRequest
		}
		projectRevision = parsed
	}
	params.ProjectRevision = projectRevision
	params.AuditRevision = revision
	if len(cursor) != 0 {
		createdAt, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			return errInvalidRequest
		}
		params.AfterCreatedAt, params.AfterFindingID = &createdAt, cursor[1]
	}
	result, err := h.Audits.ListFindingsPage(r.Context(), params)
	if err != nil {
		return err
	}
	findings := result.Items
	findings, page, err := paginate(h.cursors, findings, limit, cursorKind, func(last auditstore.ProjectFindingSummary) []string {
		return []string{
			last.CreatedAt.UTC().Format(time.RFC3339Nano),
			last.FindingID,
			strconv.FormatUint(result.ProjectRevision, 10),
			strconv.FormatUint(result.AuditRevision, 10),
		}
	})
	if err != nil {
		return err
	}
	if last := result.ContinueAfter; last != nil {
		cursor, encodeErr := h.cursors.encodePageCursor(cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID, strconv.FormatUint(result.ProjectRevision, 10), strconv.FormatUint(result.AuditRevision, 10))
		if encodeErr != nil {
			return encodeErr
		}
		page = pageInfoResponse{HasMore: true, NextCursor: &cursor}
	}
	writeJSON(w, http.StatusOK, findingPageResponse{Items: findings, Page: page, FindingPageBasis: result.FindingPageBasis})
	return nil
}

func (h *auditHandlers) getAuditFinding(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	finding, err := h.Audits.GetAuditProjectFinding(
		r.Context(), principalUserID(r.Context()), r.PathValue("auditId"), r.PathValue("findingId"),
	)
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(finding.Revision, 10)))
	writeJSON(w, http.StatusOK, finding)
	return nil
}

// auditMutationHeaders returns the Idempotency-Key and the one strong If-Match
// revision that Audit lifecycle controls require.
// Callers check them before they read the body (API-R112).
func auditMutationHeaders(r *http.Request) (string, uint64, error) {
	key, err := requireIdempotencyKey(r)
	if err != nil {
		return "", 0, err
	}
	revision, err := requiredRevision(r, "Audit mutation")
	return key, revision, err
}

// Continuations bind owner, filters and Audit revision. A caller may also pin
// its first page to a summary revision; incompatible cursors are never reused.
// decodeAuditPageCursor decodes a continuation of parts values whose last
// value is the Audit revision every page of the listing is pinned to. An
// explicit auditRevision must name the same revision.
func (h *auditHandlers) decodeAuditPageCursor(value, kind, expected string, parts int) ([]string, *uint64, error) {
	var revision *uint64
	if expected != "" {
		parsed, err := strconv.ParseUint(expected, 10, 64)
		if err != nil || parsed == 0 {
			return nil, nil, errInvalidRequest
		}
		revision = &parsed
	}
	cursor, err := h.cursors.decodePageCursor(value, kind, parts)
	if err != nil {
		return nil, nil, err
	}
	if len(cursor) != 0 {
		parsed, err := strconv.ParseUint(cursor[parts-1], 10, 64)
		if err != nil || parsed == 0 || (revision != nil && *revision != parsed) {
			return nil, nil, errInvalidRequest
		}
		revision = &parsed
	}
	return cursor, revision, nil
}

func (h *auditHandlers) getAuditWorkspace(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	result, err := h.Audits.GetWorkspace(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"))
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, result)
	return nil
}
