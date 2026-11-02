package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

func (h *auditHandlers) publishAuditMaterial(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if err := requireJSONMediaType(r); err != nil {
		handleError(w, err)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	expected, err := artifactWritePrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var request auditservice.MaterialPublicationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	result, err := h.Audits.PublishMaterial(ctx, principalUserID(r.Context()), r.PathValue("projectId"), key, request, expected)
	if err != nil {
		handleError(w, err)
		return
	}
	status := http.StatusCreated
	if expected != nil || result.Replayed {
		status = http.StatusOK
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", httpx.QuotedETag(result.Artifact.Revision))
	writeJSON(w, status, result)
}
