package public

import (
	"context"
	"net/http"
	"net/url"

	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

type FindingExportManagement interface {
	PublishFindingExport(context.Context, string, string, findingexport.Request) (findingintake.PublishedExport, error)
	ReadFindingExport(context.Context, string, string, string, string) (findingintake.ExportDownload, error)
}

func (h *findingHandlers) publishAuditFindingExport(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	var request findingexport.Request
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	auditID := r.PathValue("auditId")
	if _, _, err := request.Normalize(auditID); err != nil {
		return errInvalidRequest
	}
	if _, err := h.Audits.Get(r.Context(), principalUserID(r.Context()), auditID); err != nil {
		return err
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, h.Transfers)
	if err != nil {
		return err
	}
	defer transfer.Close()
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	result, err := h.FindingExports.PublishFindingExport(ctx, principalUserID(r.Context()), auditID, request)
	transfer.ReleaseBeforeWrite(0)
	if err != nil {
		return err
	}
	w.Header().Set("Location", "/v1/audits/"+url.PathEscape(auditID)+"/findings/exports/"+url.PathEscape(result.ExportID)+"/content")
	status := http.StatusCreated
	if result.Replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
	return nil
}

func (h *findingHandlers) downloadAuditFindingExport(w http.ResponseWriter, r *http.Request) error {
	return h.downloadFindingExportPart(w, r, "export", ".sarif")
}
func (h *findingHandlers) downloadAuditFindingExportSnapshot(w http.ResponseWriter, r *http.Request) error {
	return h.downloadFindingExportPart(w, r, "export-snapshot", ".snapshot.json")
}
func (h *findingHandlers) downloadAuditFindingExportDiagnostics(w http.ResponseWriter, r *http.Request) error {
	return h.downloadFindingExportPart(w, r, "export-diagnostics", ".diagnostics.json")
}

func (h *findingHandlers) downloadFindingExportPart(w http.ResponseWriter, r *http.Request, part, suffix string) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	auditID := r.PathValue("auditId")
	if _, err := h.Audits.Get(r.Context(), principalUserID(r.Context()), auditID); err != nil {
		return err
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, h.Transfers)
	if err != nil {
		return err
	}
	defer transfer.Close()
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	download, err := h.FindingExports.ReadFindingExport(ctx, principalUserID(r.Context()), auditID, r.PathValue("exportId"), part)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		return err
	}
	read := download.Read
	transfer.BoundWrite(int64(len(read.Payload.Data)))
	writeArtifactBytes(w, read, "audit-"+auditID+suffix)
	return nil
}
