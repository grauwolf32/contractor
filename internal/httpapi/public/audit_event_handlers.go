package public

import (
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

func (h *auditHandlers) listAuditEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	_, limit, cursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	p := auditservice.EventListParams{OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: pagelimit.FetchRows(limit)}
	kind := "audit-events:" + strconv.Quote(p.OwnerID) + ":" + strconv.Quote(p.AuditID)
	position, err := h.cursors.decodePageCursor(cursor, kind, 2)
	if err == nil && len(position) != 0 {
		through, e1 := strconv.ParseUint(position[0], 10, 63)
		before, e2 := strconv.ParseUint(position[1], 10, 63)
		if e1 != nil || e2 != nil || before == 0 || before > through {
			err = errInvalidRequest
		} else {
			p.ThroughSequence, p.BeforeSequence = &through, &before
		}
	}
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.Audits.ListEventsPage(r.Context(), p)
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, result.Items, limit, kind, func(last auditservice.Event) []string {
		return []string{strconv.FormatUint(result.ThroughSequence, 10), strconv.FormatUint(last.Sequence, 10)}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if items == nil {
		items = []auditservice.Event{}
	}
	writeJSON(w, http.StatusOK, struct {
		Items           []auditservice.Event `json:"items"`
		Page            pageInfoResponse     `json:"page"`
		ThroughSequence uint64               `json:"throughSequence"`
		Total           int                  `json:"total"`
	}{items, page, result.ThroughSequence, result.Total})
}
