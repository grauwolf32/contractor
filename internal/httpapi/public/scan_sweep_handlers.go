package public

import (
	"context"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/scanservice"
	"github.com/grauwolf32/contractor/internal/scanstore"
)

type ScanSweepManagement interface {
	Create(context.Context, string, string, scanservice.Request) (scanstore.Sweep, bool, error)
	Get(context.Context, string, string) (scanstore.Sweep, error)
	List(context.Context, string, string, int) ([]scanstore.Sweep, error)
	Members(context.Context, string, string, int, int) ([]scanstore.Member, error)
	Cancel(context.Context, string, string) (scanstore.Sweep, error)
	Purge(context.Context, string, string) error
}
type ScanSweepDependencies struct{ Sweeps ScanSweepManagement }
type scanSweepHandlers struct {
	ScanSweepDependencies
	cursors pageCursors
}

func (h *scanSweepHandlers) create(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	var request scanservice.Request
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	result, replayed, err := h.Sweeps.Create(r.Context(), principalUserID(r.Context()), key, request)
	if err != nil {
		handleError(w, err)
		return
	}
	if replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result)
}
func (h *scanSweepHandlers) get(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	result, err := h.Sweeps.Get(r.Context(), principalUserID(r.Context()), r.PathValue("sweepId"))
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
func (h *scanSweepHandlers) list(w http.ResponseWriter, r *http.Request) {
	_, limit, encoded, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encoded, "scan-sweeps", 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	if len(cursor) > 0 {
		after = cursor[0]
	}
	items, err := h.Sweeps.List(r.Context(), principalUserID(r.Context()), after, pagelimit.FetchRows(limit))
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, items, limit, "scan-sweeps", func(s scanstore.Sweep) []string { return []string{s.SweepID} })
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Items []scanstore.Sweep `json:"items"`
		Page  pageInfoResponse  `json:"page"`
	}{items, page})
}
func (h *scanSweepHandlers) members(w http.ResponseWriter, r *http.Request) {
	_, limit, encoded, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	kind := "scan-sweep-members:" + r.PathValue("sweepId")
	cursor, err := h.cursors.decodePageCursor(encoded, kind, 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := -1
	if len(cursor) > 0 {
		after, err = strconv.Atoi(cursor[0])
		if err != nil {
			handleError(w, errInvalidRequest)
			return
		}
	}
	items, err := h.Sweeps.Members(r.Context(), principalUserID(r.Context()), r.PathValue("sweepId"), after, pagelimit.FetchRows(limit))
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, items, limit, kind, func(m scanstore.Member) []string { return []string{strconv.Itoa(m.Ordinal)} })
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Items []scanstore.Member `json:"items"`
		Page  pageInfoResponse   `json:"page"`
	}{items, page})
}
func (h *scanSweepHandlers) cancel(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if err := requireEmptyBody(w, r); err != nil {
		handleError(w, err)
		return
	}
	result, err := h.Sweeps.Cancel(r.Context(), principalUserID(r.Context()), r.PathValue("sweepId"))
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
func (h *scanSweepHandlers) purge(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if err := requireEmptyBody(w, r); err != nil {
		handleError(w, err)
		return
	}
	if err := h.Sweeps.Purge(r.Context(), principalUserID(r.Context()), r.PathValue("sweepId")); err != nil {
		handleError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
