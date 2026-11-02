package public

import (
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

// ArtifactScopes opens the Artifact store of one scope.
type ArtifactScopes interface {
	User(string) (artifacts.ScopedStore, error)
	Project(string) (artifacts.ScopedStore, error)
	Run(string) (artifacts.ScopedStore, error)
}

// ArtifactDependencies serve the UserScope, ProjectScope and RunScope
// Artifact routes, Run outputs, archive previews and Git import.
type ArtifactDependencies struct {
	Artifacts  ArtifactScopes
	Projects   ProjectReader
	Runs       RunGetter
	GitImports GitImportService
}

type artifactHandlers struct {
	ArtifactDependencies
	cursors pageCursors
}

func (h *artifactHandlers) listArtifacts(w http.ResponseWriter, r *http.Request) error {
	store, err := h.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		return err
	}
	return h.listArtifactBindings(
		w, r, store, artifactListOptions{
			uploadsAllowed: true, cursorPrefix: "user-artifacts", allowNamespaceExclusion: true,
			hiddenNamespace: artifactpolicy.AuditStandardCatalogNamespace,
		},
	)
}

func (h *artifactHandlers) getArtifactMetadata(w http.ResponseWriter, r *http.Request) error {
	if err := validateUserArtifactReadRoute(r); err != nil {
		return err
	}
	store, err := h.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		return err
	}
	return h.getArtifactMetadataFromStore(w, r, store, true)
}

func (h *artifactHandlers) listArtifactVersions(w http.ResponseWriter, r *http.Request) error {
	if err := validateUserArtifactReadRoute(r); err != nil {
		return err
	}
	store, err := h.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		return err
	}
	return h.listArtifactVersionsFromStore(w, r, store, true, "user-artifact-versions")
}

func (h *artifactHandlers) listArtifactLineage(w http.ResponseWriter, r *http.Request) error {
	if err := validateUserArtifactReadRoute(r); err != nil {
		return err
	}
	store, err := h.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		return err
	}
	return h.listArtifactLineageFromStore(w, r, store, "user-artifact-lineage")
}

func (h *artifactHandlers) listRunArtifacts(w http.ResponseWriter, r *http.Request) error {
	store, runID, err := h.ownedRunArtifactStore(r)
	if err != nil {
		return err
	}
	return h.listArtifactBindings(
		w, r, store, artifactListOptions{cursorPrefix: "run-artifacts:" + runID},
	)
}

func (h *artifactHandlers) getRunArtifact(w http.ResponseWriter, r *http.Request) error {
	store, _, err := h.ownedRunArtifactStore(r)
	if err != nil {
		return err
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		return err
	}
	return h.writeArtifactRead(w, r, store, ref)
}

func (h *artifactHandlers) getRunArtifactMetadata(w http.ResponseWriter, r *http.Request) error {
	store, _, err := h.ownedRunArtifactStore(r)
	if err != nil {
		return err
	}
	return h.getArtifactMetadataFromStore(w, r, store, false)
}

func (h *artifactHandlers) listRunArtifactVersions(w http.ResponseWriter, r *http.Request) error {
	store, runID, err := h.ownedRunArtifactStore(r)
	if err != nil {
		return err
	}
	return h.listArtifactVersionsFromStore(w, r, store, false, "run-artifact-versions:"+runID)
}

func (h *artifactHandlers) listRunArtifactLineage(w http.ResponseWriter, r *http.Request) error {
	store, runID, err := h.ownedRunArtifactStore(r)
	if err != nil {
		return err
	}
	return h.listArtifactLineageFromStore(w, r, store, "run-artifact-lineage:"+runID)
}

func (h *artifactHandlers) ownedRunArtifactStore(r *http.Request) (artifacts.ScopedStore, string, error) {
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		return artifacts.ScopedStore{}, "", err
	}
	store, err := h.Artifacts.Run(run.RunID)
	return store, run.RunID, err
}

type artifactListOptions struct {
	uploadsAllowed          bool
	cursorPrefix            string
	allowNamespaceExclusion bool
	hiddenNamespace         string
	excludeNamespacePrefix  string
}

func (h *artifactHandlers) listArtifactBindings(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, options artifactListOptions,
) error {
	allowedQuery := []string{"namespace"}
	if options.allowNamespaceExclusion {
		allowedQuery = append(allowedQuery, "excludeNamespace")
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, allowedQuery...)
	if err != nil {
		return err
	}
	var namespace *string
	if values, present := query["namespace"]; present {
		value := values[0]
		if err := validatePublicArtifactName(value); err != nil {
			return err
		}
		namespace = &value
	}
	cursorKind := options.cursorPrefix
	if namespace != nil {
		cursorKind += ":" + *namespace
	}
	var excludeNamespace *string
	if values, present := query["excludeNamespace"]; present {
		value := values[0]
		if err := validatePublicArtifactName(value); err != nil {
			return err
		}
		excludeNamespace = &value
		cursorKind += ":exclude:" + value
	}
	if options.hiddenNamespace != "" {
		cursorKind += ":hidden:" + options.hiddenNamespace
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		return err
	}
	pageQuery := artifacts.BindingPageQuery{
		Namespace: namespace, ExcludeNamespace: excludeNamespace, HiddenNamespace: options.hiddenNamespace,
		ExcludeNamespacePrefix: options.excludeNamespacePrefix, Limit: pagelimit.FetchRows(limit),
	}
	if len(cursor) != 0 {
		pageQuery.AfterNamespace, pageQuery.AfterName = cursor[0], cursor[1]
	}
	items, err := store.ListMetadata(r.Context(), pageQuery)
	if err != nil {
		return err
	}
	items, page, err := paginate(h.cursors, items, limit, cursorKind, func(last artifacts.Metadata) []string {
		return []string{last.Ref.Namespace, last.Ref.Name}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, artifactPageResponse{Items: ownerArtifactPage(store, items, options.uploadsAllowed), Page: page})
	return nil
}

func (h *artifactHandlers) getArtifactMetadataFromStore(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, uploadsAllowed bool,
) error {
	ref, err := artifactRouteRef(r)
	if err != nil {
		return err
	}
	metadata, err := store.Metadata(r.Context(), ref)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, ownerArtifactMetadata(store, metadata, uploadsAllowed))
	return nil
}

func (h *artifactHandlers) listArtifactVersionsFromStore(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, uploadsAllowed bool, cursorPrefix string,
) error {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if err := validateArtifactRouteNames(r); err != nil {
		return err
	}
	cursorKind := cursorPrefix + ":" + ref.Namespace + ":" + ref.Name
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		return err
	}
	pageQuery := artifacts.VersionPageQuery{Limit: pagelimit.FetchRows(limit)}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			return fmt.Errorf("%w: invalid Artifact cursor", errInvalidRequest)
		}
		pageQuery.BeforeCreatedAt = &before
		pageQuery.BeforeRevision = cursor[1]
	}
	items, err := store.ListVersions(r.Context(), ref, pageQuery)
	if err != nil {
		return err
	}
	items, page, err := paginate(h.cursors, items, limit, cursorKind, func(last artifacts.Metadata) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), *last.Ref.Revision}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, artifactPageResponse{Items: ownerArtifactPage(store, items, uploadsAllowed), Page: page})
	return nil
}

func (h *artifactHandlers) listArtifactLineageFromStore(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, cursorPrefix string,
) error {
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "revision")
	if err != nil {
		return err
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if err := validateArtifactRouteNames(r); err != nil {
		return err
	}
	if values, present := query["revision"]; present {
		if err := validatePublicRevision(values[0]); err != nil {
			return err
		}
		ref.Revision = &values[0]
	}
	cursorKind := cursorPrefix + ":" + ref.Namespace + ":" + ref.Name
	if ref.Revision != nil {
		cursorKind += ":" + *ref.Revision
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 5)
	if err != nil {
		return err
	}
	selectedRevision := ""
	if len(cursor) != 0 {
		selectedRevision = cursor[0]
		if ref.Revision != nil && *ref.Revision != selectedRevision {
			return fmt.Errorf("%w: Artifact cursor selects another revision", errInvalidRequest)
		}
		ref.Revision = &selectedRevision
	} else {
		metadata, metadataErr := store.Metadata(r.Context(), ref)
		if metadataErr != nil {
			return metadataErr
		}
		selectedRevision = *metadata.Ref.Revision
		ref = metadata.Ref
	}
	pageQuery := artifacts.LineagePageQuery{
		ExcludeKind: artifacts.LineageAuditImport,
		Limit:       pagelimit.FetchRows(limit),
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[1])
		if parseErr != nil {
			return fmt.Errorf("%w: invalid Artifact cursor", errInvalidRequest)
		}
		pageQuery.BeforeCreatedAt = &before
		pageQuery.BeforeTargetRevision = cursor[2]
		pageQuery.BeforeSourceRevision = cursor[3]
		pageQuery.BeforeKind = cursor[4]
	}
	items, err := store.ListLineage(r.Context(), ref, pageQuery)
	if err != nil {
		return err
	}
	items, page, err := paginate(h.cursors, items, limit, cursorKind, func(last artifacts.LineageEdge) []string {
		return []string{
			selectedRevision,
			last.CreatedAt.UTC().Format(time.RFC3339Nano),
			*last.Target.Revision,
			*last.Source.Revision,
			last.Kind,
		}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, artifactLineagePageResponse{Items: items, Page: page})
	return nil
}

func artifactRouteRef(r *http.Request) (contracts.ArtifactRef, error) {
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if err := validateArtifactRouteNames(r); err != nil {
		return contracts.ArtifactRef{}, err
	}
	if values, present := query["revision"]; present {
		if err := validatePublicRevision(values[0]); err != nil {
			return contracts.ArtifactRef{}, err
		}
		ref.Revision = &values[0]
	}
	return ref, nil
}

func validateArtifactRouteNames(r *http.Request) error {
	if err := validatePublicArtifactName(r.PathValue("namespace")); err != nil {
		return err
	}
	return validatePublicArtifactName(r.PathValue("name"))
}

func (h *artifactHandlers) putArtifact(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := validateArtifactRouteNames(r); err != nil {
		return err
	}
	if artifactpolicy.IsAuditStandardCatalogNamespace(r.PathValue("namespace")) {
		return artifacts.ErrReservedNamespace
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		return err
	}
	expectedRevision, err := artifactWritePrecondition(r)
	if err != nil {
		return err
	}
	store, err := h.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		return err
	}
	return h.writeArtifactUpload(w, r, store, mediaType, expectedRevision)
}

// writeArtifactUpload holds one of the four per-process transfer slots from
// before the body is buffered until the payload is stored. Requests rejected
// by validation or ownership never take one, and the small result is written
// after the slot is released.
func (h *artifactHandlers) writeArtifactUpload(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, mediaType string, expectedRevision *string,
) error {
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		return err
	}
	defer transfer.Close()
	payload, err := readArtifactBody(r, transfer)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		return err
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	result, err := store.Write(ctx, contracts.ArtifactRef{
		Namespace: r.PathValue("namespace"), Name: r.PathValue("name"),
	}, artifacts.Payload{MediaType: mediaType, Data: payload}, expectedRevision)
	transfer.ReleaseBeforeWrite(0)
	if err != nil {
		return err
	}
	status := http.StatusCreated
	if expectedRevision != nil {
		status = http.StatusOK
	}
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	writeJSON(w, status, artifactWriteResponse{
		Artifact: result.Ref, MediaType: result.MediaType, Size: result.Size,
	})
	return nil
}

func (h *artifactHandlers) getArtifact(w http.ResponseWriter, r *http.Request) error {
	if err := validateUserArtifactReadRoute(r); err != nil {
		return err
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		return err
	}
	store, err := h.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		return err
	}
	return h.writeArtifactRead(w, r, store, ref)
}

// writeArtifactRead holds one of the four per-process transfer slots only while the
// payload is read and written, after method, ownership and route checks have
// passed, so rejected requests cannot exhaust transfer capacity.
func (h *artifactHandlers) writeArtifactRead(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, ref contracts.ArtifactRef,
) error {
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		return err
	}
	defer transfer.Close()
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	result, err := store.Read(ctx, ref)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		return err
	}
	transfer.BoundWrite(int64(len(result.Payload.Data)))
	writeArtifactBytes(w, result, "")
	return nil
}

func validateUserArtifactReadRoute(r *http.Request) error {
	if err := validateArtifactRouteNames(r); err != nil {
		return err
	}
	if artifactpolicy.IsAuditStandardCatalogNamespace(r.PathValue("namespace")) {
		return artifacts.ErrArtifactNotFound
	}
	return nil
}

func (h *artifactHandlers) getRunOutput(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	run, err := ownedRun(r, h.Runs)
	if err != nil {
		return err
	}
	store, err := h.Artifacts.Run(run.RunID)
	if err != nil {
		return err
	}
	return h.writeArtifactRead(w, r, store, contracts.ArtifactRef{Namespace: "outputs", Name: r.PathValue("slot")})
}
