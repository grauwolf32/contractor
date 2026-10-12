package privateartifacts

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

const (
	bindingCreatedAtHeader  = "X-Contractor-Binding-Created-At"
	revisionCreatedAtHeader = "X-Contractor-Revision-Created-At"
)

// listArtifacts serves the two bounded listing forms of ART-R47. Without
// namePrefix it returns one keyset page of model-visible bindings: the store
// excludes Run system and model-hidden bindings before the limit, so a page is
// never shortened, and nextCursor continues the traversal. With namePrefix it
// returns at most limit refs of one Namespace for trusted adapters (Memory).
func (h *handler) listArtifacts(w http.ResponseWriter, r *http.Request) {
	query, err := exactQuery(r.URL.RawQuery, "namespace", "namePrefix", "limit", "cursor")
	if err != nil {
		h.handleError(w, err)
		return
	}
	limitValues, limitPresent := query["limit"]
	if !limitPresent {
		h.handleError(w, fmt.Errorf("%w: limit is required", errInvalidRequest))
		return
	}
	limit, err := strconv.Atoi(limitValues[0])
	if err != nil {
		h.handleError(w, fmt.Errorf("%w: limit must be an integer", errInvalidRequest))
		return
	}
	var namespace *string
	if values, present := query["namespace"]; present {
		if values[0] == "" {
			h.handleError(w, fmt.Errorf("%w: namespace must not be empty", errInvalidRequest))
			return
		}
		namespace = &values[0]
	}
	prefixValues, prefixPresent := query["namePrefix"]
	cursorValues, cursorPresent := query["cursor"]
	if prefixPresent && (namespace == nil || cursorPresent) {
		h.handleError(w, fmt.Errorf("%w: namePrefix requires namespace and takes no cursor", errInvalidRequest))
		return
	}
	pageQuery := artifacts.RefPageQuery{Namespace: namespace, ModelVisible: true, Limit: limit}
	if cursorPresent {
		cursor, decodeErr := decodeListCursor(cursorValues[0])
		if decodeErr != nil {
			h.handleError(w, decodeErr)
			return
		}
		pageQuery.AfterNamespace, pageQuery.AfterName = cursor.Namespace, cursor.Name
	}
	store, err := h.runStore(r, false)
	if err != nil {
		h.handleError(w, err)
		return
	}
	response := contracts.ArtifactListResult{APIVersion: contracts.APIVersion}
	if prefixPresent {
		refs, listErr := store.ListPrefix(r.Context(), *namespace, prefixValues[0], limit)
		if listErr != nil {
			h.handleError(w, listErr)
			return
		}
		if artifactpolicy.IsRunSystemNamespace(*namespace) {
			refs = refs[:0]
		}
		response.Artifacts = refs
	} else {
		page, listErr := store.ListRefs(r.Context(), pageQuery)
		if listErr != nil {
			h.handleError(w, listErr)
			return
		}
		response.Artifacts = page.Refs
		if page.More {
			next, encodeErr := encodeListCursor(page.Refs[len(page.Refs)-1])
			if encodeErr != nil {
				h.handleError(w, encodeErr)
				return
			}
			response.NextCursor = &next
		}
	}
	if err := response.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

// getArtifact takes a transfer slot only after the request and the allocation
// grant are accepted, and holds it while the payload is read and written.
func (h *handler) getArtifact(w http.ResponseWriter, r *http.Request) {
	store, ref, err := h.readTarget(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	result, err := store.Read(ctx, ref)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	transfer.BoundWrite(int64(len(result.Payload.Data)))
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	setArtifactTimestampHeaders(w.Header(), result.BindingCreatedAt, result.RevisionCreatedAt)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}

// getArtifactMetadata answers what the raw read of the same ref would return,
// its exact revision, media type and size, without reading or sending the
// payload, so a reader can refuse or reuse content before downloading it
// (ART-R116).
func (h *handler) getArtifactMetadata(w http.ResponseWriter, r *http.Request) {
	store, ref, err := h.readTarget(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		h.handleError(w, err)
		return
	}
	response := contracts.ArtifactReadResult{
		APIVersion: contracts.APIVersion, Artifact: metadata.Ref,
		MediaType: metadata.MediaType, Size: metadata.Size,
	}
	if err := response.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

// readTarget validates a raw or metadata read of one current or exact ref and
// resolves the allocation's RunScope for it.
func (h *handler) readTarget(r *http.Request) (artifacts.ScopedStore, contracts.ArtifactRef, error) {
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		return artifacts.ScopedStore{}, contracts.ArtifactRef{}, err
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if values, present := query["revision"]; present {
		if values[0] == "" {
			return artifacts.ScopedStore{}, contracts.ArtifactRef{},
				fmt.Errorf("%w: revision must not be empty", errInvalidRequest)
		}
		ref.Revision = &values[0]
	}
	store, err := h.runStore(r, false)
	if err != nil {
		return artifacts.ScopedStore{}, contracts.ArtifactRef{}, err
	}
	if artifactpolicy.IsRunSystemNamespace(ref.Namespace) {
		return artifacts.ScopedStore{}, contracts.ArtifactRef{}, artifacts.ErrReservedNamespace
	}
	return store, ref, nil
}

// putArtifact takes a transfer slot after the request and allocation checks,
// before buffering the body, and releases it before writing the small result.
func (h *handler) putArtifact(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	expectedRevision, create, err := artifactWritePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	// Reject unknown/fenced allocations before accepting a large payload. The
	// second lookup below is the write authorization linearization point and
	// catches a fence established while the request body was arriving.
	if _, err := h.runStore(r, true); err != nil {
		h.handleError(w, err)
		return
	}
	if artifactpolicy.IsRunSystemNamespace(r.PathValue("namespace")) {
		h.handleError(w, artifacts.ErrReservedNamespace)
		return
	}
	allocationID := r.PathValue("allocationID")
	identity, ok := r.Context().Value(authenticatedRuntimeContextKey{}).(authenticatedRuntime)
	if !ok {
		h.handleError(w, errArtifactAccessDenied)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	payload, err := readArtifactBody(r, transfer)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	preparedPayload, err := artifacts.PreparePayload(ctx, artifacts.Payload{MediaType: mediaType, Data: payload})
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	var result artifacts.WriteResult
	err = h.dependencies.Registry.WithWriteGrant(
		allocationID,
		func(grant controlplane.AllocationGrant) error {
			store, storeErr := h.runStoreForGrant(allocationID, grant, identity, true)
			if storeErr != nil {
				return storeErr
			}
			result, storeErr = store.Write(
				ctx,
				contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")},
				preparedPayload,
				expectedRevision,
			)
			return storeErr
		},
	)
	transfer.ReleaseBeforeWrite(0)
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusOK
	if create {
		status = http.StatusCreated
	}
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	setArtifactTimestampHeaders(w.Header(), result.BindingCreatedAt, result.RevisionCreatedAt)
	writeJSON(w, status, contracts.ArtifactWriteResult{
		APIVersion: contracts.APIVersion, Artifact: result.Ref,
		MediaType: result.MediaType, Size: result.Size,
	})
}

func setArtifactTimestampHeaders(header http.Header, bindingCreatedAt, revisionCreatedAt time.Time) {
	header.Set(bindingCreatedAtHeader, bindingCreatedAt.UTC().Format(time.RFC3339Nano))
	header.Set(revisionCreatedAtHeader, revisionCreatedAt.UTC().Format(time.RFC3339Nano))
}
