package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

func (h *auditHandlers) getAuditArtifact(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, err := exactQuery(r.URL.RawQuery, "key", "revision", "member")
	if err != nil || !validAuditArtifactSelector(query.Get("key"), query.Get("revision"), query.Get("member")) {
		return errInvalidRequest
	}
	if _, err := h.Audits.Get(r.Context(), principalUserID(r.Context()), r.PathValue("auditId")); err != nil {
		return err
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, h.Transfers)
	if err != nil {
		return err
	}
	defer transfer.Close()
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	read, err := h.Audits.ReadArtifact(ctx, principalUserID(r.Context()), r.PathValue("auditId"), query.Get("key"), query.Get("revision"))
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		return err
	}
	if id := query.Get("member"); id != "" {
		pkg, err := auditdomain.ValidatePackage(read.Payload.Data)
		if err != nil {
			transfer.ReleaseBeforeWrite(0)
			return artifacts.ErrArtifactNotFound
		}
		member, exists := pkg.MemberByID(id)
		if !exists {
			transfer.ReleaseBeforeWrite(0)
			return artifacts.ErrArtifactNotFound
		}
		read.Payload = artifacts.Payload{MediaType: member.Metadata().MediaType, Data: member.Data()}
	}
	transfer.BoundWrite(int64(len(read.Payload.Data)))
	writeArtifactBytes(w, read, "")
	return nil
}

// validAuditArtifactSelector bounds the selector of one Audit artifact read
// (ASC-R148): a logical key of 1 to 512 UTF-8 bytes (LIM-AUD-39), a revision
// of 1 to 256 bytes and an optional member ID of at most 128 bytes. Revisions
// and member IDs are ASCII by their grammars, which the public schema
// declares; a value outside them matches nothing and reads as 404.
func validAuditArtifactSelector(key, revision, member string) bool {
	return key != "" && len(key) <= auditstore.MaxArtifactLogicalKeyBytes &&
		revision != "" && len(revision) <= 256 && len(member) <= 128
}
