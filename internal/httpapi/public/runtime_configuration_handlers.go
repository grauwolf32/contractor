package public

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const maximumRuntimeConfigBodyBytes int64 = 128 * 1024

var (
	runtimeConfigIDPattern         = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,62}$`)
	runtimeConfigVersionPattern    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$`)
	runtimeCredentialIDPattern     = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,127}$`)
	runtimeAgentPrincipalIDPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func (h *handler) listRuntimeConfigs(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, "runtime-configs", 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	afterName, afterVersion := "", ""
	if len(cursor) != 0 {
		afterName, afterVersion = cursor[0], cursor[1]
	}
	versions, err := h.dependencies.RuntimeConfigs.ListVersions(r.Context(), afterName, afterVersion, limit+1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(versions) > limit {
		versions = versions[:limit]
		last := versions[len(versions)-1].Ref
		next, cursorErr := h.encodePageCursor("runtime-configs", last.Name, last.Version)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	items := make([]runtimeConfigResourceResponse, len(versions))
	for index := range versions {
		items[index] = runtimeConfigResource(versions[index])
	}
	writeJSON(w, http.StatusOK, runtimeConfigPageResponse{Items: items, Page: page})
}

func (h *handler) getRuntimeConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	name, version := r.PathValue("name"), r.PathValue("version")
	if !runtimeConfigIDPattern.MatchString(name) || !runtimeConfigVersionPattern.MatchString(version) {
		h.handleError(w, errInvalidRequest)
		return
	}
	resource, err := h.dependencies.RuntimeConfigs.GetVersion(r.Context(), name, version)
	if err != nil {
		if errors.Is(err, runtimeconfig.ErrNotFound) {
			h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
			return
		}
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(resource.Ref.Digest))
	writeJSON(w, http.StatusOK, runtimeConfigResource(resource))
}

func (h *handler) publishRuntimeConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	body, err := readBoundedRuntimeJSON(w, r, maximumRuntimeConfigBodyBytes)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer wipePublicBytes(body)
	result, err := h.dependencies.RuntimeConfigs.Publish(
		r.Context(), body, idempotencyKey, principalUserID(r.Context()),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, result.Version.Ref.Name,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_config.publish", result.Version.Ref.Name, result.Replayed)
	w.Header().Set("ETag", strconv.Quote(result.Version.Ref.Digest))
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, runtimeConfigResource(result.Version))
}

func (h *handler) listRuntimeLabels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, "runtime-labels", 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	bindings, err := h.dependencies.RuntimeConfigs.ListBindings(r.Context(), after, limit+1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(bindings) > limit {
		bindings = bindings[:limit]
		next, cursorErr := h.encodePageCursor("runtime-labels", bindings[len(bindings)-1].Label)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	items := make([]runtimeLabelResponse, len(bindings))
	for index := range bindings {
		items[index] = runtimeLabelResource(bindings[index])
	}
	writeJSON(w, http.StatusOK, runtimeLabelPageResponse{Items: items, Page: page})
}

func (h *handler) getRuntimeLabel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	label := r.PathValue("label")
	if !runtimeConfigIDPattern.MatchString(label) {
		h.handleError(w, errInvalidRequest)
		return
	}
	binding, err := h.dependencies.RuntimeConfigs.GetBinding(r.Context(), label)
	if err != nil {
		if errors.Is(err, runtimeconfig.ErrNotFound) {
			h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
			return
		}
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(binding.Revision, 10)))
	writeJSON(w, http.StatusOK, runtimeLabelResource(binding))
}

func (h *handler) putRuntimeLabel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	label := r.PathValue("label")
	if !runtimeConfigIDPattern.MatchString(label) {
		h.handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	mode, expectedRevision, err := runtimeLabelPutPrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request runtimeLabelMutationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	actor := principalUserID(r.Context())
	var result runtimeconfig.BindingMutationResult
	if mode == labelMutationCreate {
		result, err = h.dependencies.RuntimeConfigs.CreateBinding(
			r.Context(), label, request.Config, idempotencyKey, actor, h.dependencies.Now(),
		)
	} else {
		result, err = h.dependencies.RuntimeConfigs.Rebind(
			r.Context(), label, expectedRevision, request.Config, idempotencyKey, actor, h.dependencies.Now(),
		)
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	if result.Binding == nil {
		h.handleError(w, errors.New("Runtime label mutation returned no binding"))
		return
	}
	if !result.Replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, label,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_label.put", label, result.Replayed)
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Binding.Revision, 10)))
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	status := http.StatusOK
	if mode == labelMutationCreate {
		status = http.StatusCreated
	}
	writeJSON(w, status, runtimeLabelResource(*result.Binding))
}

func (h *handler) deleteRuntimeLabel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	label := r.PathValue("label")
	if !runtimeConfigIDPattern.MatchString(label) {
		h.handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	expectedRevision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := h.dependencies.RuntimeConfigs.DeleteBinding(
		r.Context(), label, expectedRevision, idempotencyKey,
		principalUserID(r.Context()), h.dependencies.Now(),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !result.Deleted {
		h.handleError(w, errors.New("Runtime label mutation did not delete the binding"))
		return
	}
	if !result.Replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, label,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_label.delete", label, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *handler) listRuntimeCredentials(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, "runtime-credentials", 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	items, err := h.dependencies.RuntimeCredentials.List(r.Context(), after, limit+1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		next, cursorErr := h.encodePageCursor("runtime-credentials", items[len(items)-1].CredentialID)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, runtimeCredentialPageResponse{Items: items, Page: page})
}

func (h *handler) getRuntimeCredential(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	id := r.PathValue("credentialId")
	if !runtimeCredentialIDPattern.MatchString(id) {
		h.handleError(w, errInvalidRequest)
		return
	}
	metadata, err := h.dependencies.RuntimeCredentials.Get(r.Context(), id)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, metadata)
}

func (h *handler) createRuntimeCredential(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request createRuntimeCredentialRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	defer request.Material.Destroy()
	if !runtimeCredentialIDPattern.MatchString(request.CredentialID) {
		h.handleError(w, errInvalidRequest)
		return
	}
	result, err := h.dependencies.RuntimeCredentials.Create(r.Context(), credentials.RuntimeCredentialCreateRequest{
		CredentialID: request.CredentialID, Material: request.Material,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, "runtime:"+result.Credential.CredentialID,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_credential.create", result.Credential.CredentialID, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Credential)
}

func (h *handler) deleteRuntimeCredential(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	id := r.PathValue("credentialId")
	if !runtimeCredentialIDPattern.MatchString(id) {
		h.handleError(w, errInvalidRequest)
		return
	}
	if _, err := requireIdempotencyKey(r); err != nil {
		h.handleError(w, err)
		return
	}
	result, err := h.dependencies.RuntimeCredentials.Delete(
		r.Context(), id, principalUserID(r.Context()),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, "runtime:"+id,
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_credential.delete", id, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
}

func runtimeConfigResource(version runtimeconfig.Version) runtimeConfigResourceResponse {
	return runtimeConfigResourceResponse{
		Ref: version.Ref, Document: append([]byte(nil), version.CanonicalDocument...),
		BuiltIn: version.BuiltIn, CreatedBy: version.ActorID, CreatedAt: version.CreatedAt,
	}
}

func runtimeLabelResource(binding runtimeconfig.Binding) runtimeLabelResponse {
	return runtimeLabelResponse{
		Label: binding.Label, Config: binding.Ref, Revision: strconv.FormatUint(binding.Revision, 10),
		CreatedBy: binding.CreatedBy, CreatedAt: binding.CreatedAt,
		UpdatedBy: binding.UpdatedBy, UpdatedAt: binding.UpdatedAt,
	}
}

type runtimeLabelMutationMode uint8

const (
	labelMutationCreate runtimeLabelMutationMode = iota + 1
	labelMutationRebind
)

func runtimeLabelPutPrecondition(r *http.Request) (runtimeLabelMutationMode, uint64, error) {
	ifMatch, ifNoneMatch := r.Header.Values("If-Match"), r.Header.Values("If-None-Match")
	if len(ifNoneMatch) == 1 && strings.TrimSpace(ifNoneMatch[0]) == "*" && len(ifMatch) == 0 {
		return labelMutationCreate, 0, nil
	}
	if len(ifNoneMatch) != 0 || len(ifMatch) != 1 {
		return 0, 0, fmt.Errorf("%w: binding PUT requires If-None-Match: * or one If-Match", errInvalidRequest)
	}
	revision, err := parseRuntimeRevisionETag(ifMatch[0])
	if err != nil {
		return 0, 0, err
	}
	return labelMutationRebind, revision, nil
}

func runtimeLabelDeletePrecondition(r *http.Request) (uint64, error) {
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		return 0, fmt.Errorf("%w: binding DELETE requires one If-Match", errInvalidRequest)
	}
	return parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
}

func parseRuntimeRevisionETag(raw string) (uint64, error) {
	value := strings.TrimSpace(raw)
	if strings.HasPrefix(value, "W/") || strings.Contains(value, ",") {
		return 0, fmt.Errorf("%w: binding revision requires one strong ETag", errInvalidRequest)
	}
	revision, err := strconv.Unquote(value)
	if err != nil || revision == "" {
		return 0, fmt.Errorf("%w: binding revision must be quoted", errInvalidRequest)
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil || parsed == 0 || strconv.FormatUint(parsed, 10) != revision {
		return 0, fmt.Errorf("%w: binding revision is invalid", errInvalidRequest)
	}
	return parsed, nil
}

func readBoundedRuntimeJSON(w http.ResponseWriter, r *http.Request, maximum int64) ([]byte, error) {
	if maximum <= 0 || r.ContentLength > maximum {
		return nil, fmt.Errorf("%w: JSON body is too large", errInvalidRequest)
	}
	body := http.MaxBytesReader(w, r.Body, maximum)
	data, err := io.ReadAll(body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return nil, fmt.Errorf("%w: JSON body is too large", errInvalidRequest)
		}
		return nil, errors.New("read RuntimeConfig request")
	}
	if len(data) == 0 {
		return nil, errInvalidRequest
	}
	return data, nil
}

func (h *handler) auditRuntimeMutation(r *http.Request, action, resourceID string, replayed bool) {
	if h.dependencies.Logger == nil {
		return
	}
	h.dependencies.Logger.InfoContext(
		r.Context(), "Runtime Operations mutation",
		"audit_action", action,
		"actor_id", principalUserID(r.Context()),
		"request_id", requestid.From(r.Context()),
		"resource_id", resourceID,
		"replayed", replayed,
	)
}
