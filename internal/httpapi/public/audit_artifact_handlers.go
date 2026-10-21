package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

func (h *auditHandlers) getAuditArtifact(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	query, err := exactQuery(r.URL.RawQuery, "key", "revision", "member")
	if err != nil || query.Get("key") == "" || len(query.Get("key")) > 512 || query.Get("revision") == "" || len(query.Get("revision")) > 256 || len(query.Get("member")) > 128 {
		handleError(w, errInvalidRequest)
		return
	}
	if _, err := h.Audits.Get(r.Context(), principalUserID(r.Context()), r.PathValue("auditId")); err != nil {
		handleError(w, err)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		handleError(w, err)
		return
	}
	defer transfer.Close()
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	read, err := h.Audits.ReadArtifact(ctx, principalUserID(r.Context()), r.PathValue("auditId"), query.Get("key"), query.Get("revision"))
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		handleError(w, err)
		return
	}
	if id := query.Get("member"); id != "" {
		pkg, err := auditdomain.ValidatePackage(read.Payload.Data)
		if err != nil {
			transfer.ReleaseBeforeWrite(0)
			handleError(w, artifacts.ErrArtifactNotFound)
			return
		}
		member, exists := pkg.MemberByID(id)
		if !exists {
			transfer.ReleaseBeforeWrite(0)
			handleError(w, artifacts.ErrArtifactNotFound)
			return
		}
		read.Payload = artifacts.Payload{MediaType: member.Metadata().MediaType, Data: member.Data()}
	}
	transfer.BoundWrite(int64(len(read.Payload.Data)))
	writeArtifactBytes(w, read)
}
