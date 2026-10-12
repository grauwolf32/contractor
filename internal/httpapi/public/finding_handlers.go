package public

import (
	"context"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

type FindingProposalManagement interface {
	ListRun(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	ListAuditInbox(context.Context, string, string, findingintake.ListQuery) ([]findingintake.Receipt, error)
	ImportIntoAudit(context.Context, findingintake.ImportRequest) (findingintake.AuditHold, bool, error)
}

// AuditReader reads one owner's Audit.
type AuditReader interface {
	Get(context.Context, string, string) (auditstore.Audit, error)
}

// FindingDependencies serve the finding proposals of Runs and Audits, their
// import into an Audit and finding collection publication.
type FindingDependencies struct {
	FindingProposals   FindingProposalManagement
	FindingCollections FindingCollectionManagement
	Audits             AuditReader
	Runs               RunGetter
}

type findingHandlers struct {
	FindingDependencies
	cursors pageCursors
}

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

func (h *findingHandlers) listRunFindingProposals(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		handleError(w, err)
		return
	}
	h.listFindingProposals(
		w, r, "run-findings:"+run.RunID,
		func(query findingintake.ListQuery) ([]findingintake.Receipt, error) {
			return h.FindingProposals.ListRun(
				r.Context(), principalUserID(r.Context()), run.RunID, query,
			)
		},
	)
}

func (h *findingHandlers) listAuditFindingProposals(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	auditID := r.PathValue("auditId")
	if _, err := h.Audits.Get(
		r.Context(), principalUserID(r.Context()), auditID,
	); err != nil {
		handleError(w, err)
		return
	}
	h.listFindingProposals(
		w, r, "audit-findings:"+auditID,
		func(query findingintake.ListQuery) ([]findingintake.Receipt, error) {
			return h.FindingProposals.ListAuditInbox(
				r.Context(), principalUserID(r.Context()), auditID, query,
			)
		},
	)
}

func (h *findingHandlers) listFindingProposals(
	w http.ResponseWriter,
	r *http.Request,
	cursorKind string,
	list func(findingintake.ListQuery) ([]findingintake.Receipt, error),
) {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		handleError(w, err)
		return
	}
	query := findingintake.ListQuery{Limit: pagelimit.FetchRows(limit)}
	if len(cursor) != 0 {
		after, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			handleError(w, errInvalidRequest)
			return
		}
		query.AfterCreatedAt, query.AfterReceiptID = &after, cursor[1]
	}
	items, err := list(query)
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, items, limit, cursorKind, func(last findingintake.Receipt) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.ReceiptID}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, findingProposalPageResponse{Items: items, Page: page})
}

func (h *findingHandlers) importAuditFindingProposal(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	var request importFindingProposalRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	hold, replayed, err := h.FindingProposals.ImportIntoAudit(
		r.Context(), findingintake.ImportRequest{
			OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"),
			RunID: request.RunID, Proposal: request.Proposal,
		},
	)
	if err != nil {
		handleError(w, err)
		return
	}
	status := http.StatusCreated
	if replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, importFindingProposalResponse{Hold: hold, Replayed: replayed})
}
