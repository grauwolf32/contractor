package public

import (
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

type projectSourcePageResponse struct {
	findingintake.ProjectSourcePage
	Page pageInfoResponse `json:"page"`
}

type projectSourceHistoryPageResponse struct {
	findingintake.ProjectSourceHistoryPage
	Page pageInfoResponse `json:"page"`
}

func (h *findingHandlers) getProjectFindingSource(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	result, err := h.FindingProposals.GetProjectReceipt(r.Context(), principalUserID(r.Context()), r.PathValue("projectId"), r.PathValue("receiptId"))
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (h *findingHandlers) listProjectFindingSources(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, encoded, err := pageQuery(r.URL.RawQuery, "findingId", "projectRevision")
	if err != nil {
		handleError(w, err)
		return
	}
	if limit > findingintake.MaxProjectSourcePage {
		handleError(w, errInvalidRequest)
		return
	}
	ownerID, projectID, findingID := principalUserID(r.Context()), r.PathValue("projectId"), query.Get("findingId")
	kind := "project-finding-sources:" + ownerID + ":" + projectID + ":" + findingID
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 3)
	if err != nil {
		handleError(w, err)
		return
	}
	selection := findingintake.ProjectSourceQuery{ListQuery: findingintake.ListQuery{Limit: limit + 1}}
	selection.ProjectRevision, err = optionalProjectRevision(query.Get("projectRevision"))
	if err != nil {
		handleError(w, err)
		return
	}
	if len(cursor) != 0 {
		after, err := time.Parse(time.RFC3339Nano, cursor[0])
		if err != nil {
			handleError(w, errInvalidRequest)
			return
		}
		selection.AfterCreatedAt, selection.AfterReceiptID = &after, cursor[1]
		revision, parseErr := projectSourceCursorRevision(cursor[2])
		if parseErr != nil || (selection.ProjectRevision != nil && *selection.ProjectRevision != *revision) {
			handleError(w, errInvalidRequest)
			return
		}
		selection.ProjectRevision = revision
	}
	result, err := h.FindingProposals.ListProjectSources(r.Context(), ownerID, projectID, findingID, selection)
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, result.Items, limit, kind, func(last findingintake.ProjectSourcePreview) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.ReceiptID, strconv.FormatUint(result.ProjectRevision, 10)}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	result.Items = items
	writeJSON(w, http.StatusOK, projectSourcePageResponse{ProjectSourcePage: result, Page: page})
}

func (h *findingHandlers) listProjectFindingSourceHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	_, limit, encoded, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	if limit > findingintake.MaxProjectSourcePage {
		handleError(w, errInvalidRequest)
		return
	}
	ownerID, projectID, receiptID := principalUserID(r.Context()), r.PathValue("projectId"), r.PathValue("receiptId")
	kind := "project-source-history:" + ownerID + ":" + projectID + ":" + receiptID
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 2)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	var revision *uint64
	if len(cursor) != 0 {
		after = cursor[0]
		revision, err = projectSourceCursorRevision(cursor[1])
		if err != nil {
			handleError(w, err)
			return
		}
	}
	result, err := h.FindingProposals.ListProjectSourceHistory(r.Context(), ownerID, projectID, receiptID, after, limit+1, revision)
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, result.Items, limit, kind, func(last artifacts.Metadata) []string {
		return []string{last.Ref.Name, strconv.FormatUint(result.ProjectRevision, 10)}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	result.Items = items
	writeJSON(w, http.StatusOK, projectSourceHistoryPageResponse{ProjectSourceHistoryPage: result, Page: page})
}
