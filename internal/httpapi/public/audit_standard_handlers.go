package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/auditstandards"
)

type auditStandardPageResponse struct {
	APIVersion string                             `json:"apiVersion"`
	Items      []auditstandards.PackageProjection `json:"items"`
}

type auditStandardResponse struct {
	APIVersion string                           `json:"apiVersion"`
	Standard   auditstandards.PackageProjection `json:"standard"`
}

func (h *handler) listAuditStandards(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	items, err := h.dependencies.Audits.Standards(r.Context(), principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, auditStandardPageResponse{
		APIVersion: "contractor/v1alpha1", Items: items,
	})
}

func (h *handler) getAuditStandard(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	standard, err := h.dependencies.Audits.Standard(
		r.Context(), principalUserID(r.Context()), auditstandards.Reference{
			Scheme: r.PathValue("scheme"), Version: r.PathValue("version"),
		},
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, auditStandardResponse{
		APIVersion: "contractor/v1alpha1", Standard: standard,
	})
}
