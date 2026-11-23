package public

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type editProjectFindingRequest struct {
	Verdict           auditstore.AnalystVerdict   `json:"verdict"`
	Severity          *auditstore.FindingSeverity `json:"severity,omitempty"`
	Rationale         string                      `json:"rationale"`
	DuplicateTargetID *string                     `json:"duplicateTargetId,omitempty"`
}

func (h *auditHandlers) getProjectFinding(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	finding, err := h.Audits.GetProjectFinding(r.Context(), principalUserID(r.Context()), r.PathValue("projectId"), r.PathValue("findingId"))
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(finding.Revision, 10)))
	writeJSON(w, http.StatusOK, finding)
	return nil
}

func (h *auditHandlers) editProjectFinding(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	key, revision, err := auditMutationHeaders(r)
	if err != nil {
		return err
	}
	var body editProjectFindingRequest
	if err := decodeJSON(w, r, &body); err != nil {
		return err
	}
	params := auditservice.ProjectFindingEditParams{ProjectFindingEdit: auditstore.ProjectFindingEdit{
		OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"), FindingID: r.PathValue("findingId"),
		ExpectedRevision: revision, Verdict: body.Verdict, Severity: body.Severity, Rationale: body.Rationale, DuplicateTargetID: body.DuplicateTargetID,
	}, IdempotencyKey: key}
	params.RequestDigest, err = auditservice.ProjectFindingEditRequestDigest(params)
	if err != nil {
		return err
	}
	result, err := h.Audits.EditProjectFinding(r.Context(), params)
	if err != nil {
		return err
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Finding.Revision, 10)))
	writeJSON(w, http.StatusOK, result)
	return nil
}

func (h *auditHandlers) listProjectFindings(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encoded, err := pageQuery(r.URL.RawQuery, "projectRevision")
	if err != nil {
		return err
	}
	owner, project := principalUserID(r.Context()), r.PathValue("projectId")
	kind := "project-findings:" + owner + ":" + project
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 3)
	if err != nil {
		return err
	}
	params := auditservice.ProjectFindingListParams{OwnerFindingListParams: auditstore.OwnerFindingListParams{OwnerID: owner, ProjectID: project, Limit: limit + 1}}
	params.ProjectRevision, err = optionalProjectRevision(query.Get("projectRevision"))
	if err != nil {
		return err
	}
	if len(cursor) != 0 {
		created, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		revision, revisionErr := projectSourceCursorRevision(cursor[2])
		if parseErr != nil || revisionErr != nil || (params.ProjectRevision != nil && *params.ProjectRevision != *revision) {
			return errInvalidRequest
		}
		params.BeforeCreatedAt, params.BeforeFindingID, params.ProjectRevision = &created, cursor[1], revision
	}
	result, err := h.Audits.ListProjectFindings(r.Context(), params)
	if err != nil {
		return err
	}
	items, page, err := paginate(h.cursors, result.Items, limit, kind, func(last auditstore.ProjectFindingSummary) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID, strconv.FormatUint(result.ProjectRevision, 10)}
	})
	if err != nil {
		return err
	}
	if last := result.ContinueBefore; last != nil {
		cursor, err := h.cursors.encodePageCursor(kind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID, strconv.FormatUint(result.ProjectRevision, 10))
		if err != nil {
			return err
		}
		page = pageInfoResponse{HasMore: true, NextCursor: &cursor}
	}
	writeJSON(w, http.StatusOK, struct {
		Items           []auditstore.ProjectFindingSummary `json:"items"`
		ProjectRevision uint64                             `json:"projectRevision"`
		Page            pageInfoResponse                   `json:"page"`
	}{items, result.ProjectRevision, page})
	return nil
}

func (request *editProjectFindingRequest) UnmarshalJSON(data []byte) error {
	type plain editProjectFindingRequest
	var value plain
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := strictjson.Decode(data, &fields); err != nil {
		return err
	}
	for _, key := range []string{"severity", "duplicateTargetId"} {
		if string(fields[key]) == "null" {
			return errInvalidRequest
		}
	}
	*request = editProjectFindingRequest(value)
	return nil
}
