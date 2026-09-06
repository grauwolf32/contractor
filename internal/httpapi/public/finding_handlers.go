package public

import (
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

type findingProposalPageResponse struct {
	Items []findingintake.Receipt `json:"items"`
	Page  pageInfoResponse        `json:"page"`
}

type importFindingProposalRequest struct {
	RunID    string                `json:"runId"`
	Proposal contracts.ArtifactRef `json:"proposal"`
}

type importFindingProposalResponse struct {
	Hold     findingintake.AuditHold `json:"hold"`
	Replayed bool                    `json:"replayed"`
}

func (h *handler) listRunFindingProposals(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.FindingProposals == nil {
		h.handleError(w, findingintake.ErrNotFound)
		return
	}
	run, err := h.ownedRun(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listFindingProposals(
		w, r, "run-findings:"+run.RunID,
		func(query findingintake.ListQuery) ([]findingintake.Receipt, error) {
			return h.dependencies.FindingProposals.ListRun(
				r.Context(), principalUserID(r.Context()), run.RunID, query,
			)
		},
	)
}

func (h *handler) listAuditFindingProposals(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.FindingProposals == nil {
		h.handleError(w, findingintake.ErrNotFound)
		return
	}
	auditID := r.PathValue("auditId")
	if _, err := h.dependencies.Audits.Get(
		r.Context(), principalUserID(r.Context()), auditID,
	); err != nil {
		h.handleError(w, err)
		return
	}
	h.listFindingProposals(
		w, r, "audit-findings:"+auditID,
		func(query findingintake.ListQuery) ([]findingintake.Receipt, error) {
			return h.dependencies.FindingProposals.ListAuditInbox(
				r.Context(), principalUserID(r.Context()), auditID, query,
			)
		},
	)
}

func (h *handler) listFindingProposals(
	w http.ResponseWriter,
	r *http.Request,
	cursorKind string,
	list func(findingintake.ListQuery) ([]findingintake.Receipt, error),
) {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	query := findingintake.ListQuery{Limit: limit + 1}
	if len(cursor) != 0 {
		after, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		query.AfterCreatedAt, query.AfterReceiptID = &after, cursor[1]
	}
	items, err := list(query)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.ReceiptID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	writeJSON(w, http.StatusOK, findingProposalPageResponse{Items: items, Page: page})
}

func (h *handler) importAuditFindingProposal(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.dependencies.FindingProposals == nil {
		h.handleError(w, findingintake.ErrNotFound)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	var request importFindingProposalRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	hold, replayed, err := h.dependencies.FindingProposals.ImportIntoAudit(
		r.Context(), findingintake.ImportRequest{
			OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"),
			RunID: request.RunID, Proposal: request.Proposal,
		},
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusCreated
	if replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, importFindingProposalResponse{Hold: hold, Replayed: replayed})
}
