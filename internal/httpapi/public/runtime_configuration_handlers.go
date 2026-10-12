package public

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const maximumRuntimeConfigBodyBytes int64 = 128 * 1024

type RuntimeConfigManagement interface {
	Publish(context.Context, []byte, string, string) (runtimeconfig.PublishResult, error)
	ListVersions(context.Context, string, string, int) ([]runtimeconfig.Version, error)
	GetVersion(context.Context, string, string) (runtimeconfig.Version, error)
	ListBindings(context.Context, string, int) ([]runtimeconfig.Binding, error)
	GetBinding(context.Context, string) (runtimeconfig.Binding, error)
	CreateBinding(context.Context, string, runtimeconfig.Ref, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
	Rebind(context.Context, string, uint64, runtimeconfig.Ref, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
	DeleteBinding(context.Context, string, uint64, string, string, time.Time) (runtimeconfig.BindingMutationResult, error)
}

type RuntimeCredentialManagement interface {
	List(context.Context, string, int) ([]credentials.RuntimeCredentialMetadata, error)
	Get(context.Context, string) (credentials.RuntimeCredentialMetadata, error)
	Create(context.Context, credentials.RuntimeCredentialCreateRequest) (credentials.RuntimeCredentialCreateResult, error)
	Delete(context.Context, string, string) (credentials.RuntimeCredentialDeleteResult, error)
}

type RuntimeAgentPrincipalManagement interface {
	List(context.Context, string, int) ([]controlplane.RuntimeAgentPrincipalProjection, bool, error)
	Get(context.Context, string) (controlplane.RuntimeAgentPrincipalProjection, error)
	ReplaceLabels(context.Context, string, uint64, []string, string, string, time.Time) (controlplane.RuntimeAgentPrincipalProjection, bool, error)
	Delete(context.Context, string, uint64, string, string, time.Time) (bool, error)
}

// RuntimeConfigurationDependencies serve RuntimeConfigs, Runtime label
// bindings, Runtime credentials and Runtime Agent principals.
type RuntimeConfigurationDependencies struct {
	RuntimeConfigs         RuntimeConfigManagement
	RuntimeCredentials     RuntimeCredentialManagement
	RuntimeAgentPrincipals RuntimeAgentPrincipalManagement
	OperationsInvalidator  OperationsInvalidator
	// Logger records every applied Runtime Operations mutation.
	Logger *slog.Logger
	Now    func() time.Time
}

type runtimeConfigurationHandlers struct {
	RuntimeConfigurationDependencies
	cursors pageCursors
}

// RuntimeConfig names and labels, versions and Runtime credential IDs use the
// contracts grammars within the bounds of the public RuntimeInfrastructureId,
// RuntimeConfigVersion and RuntimeCredentialId schemas.
func validRuntimeInfrastructureID(value string) bool {
	return len(value) <= 63 && contracts.ValidIdentifier(value)
}

func validRuntimeConfigVersion(value string) bool {
	return len(value) <= 128 && contracts.ValidVersion(value)
}

func validRuntimeCredentialID(value string) bool {
	return len(value) <= 128 && contracts.ValidIdentifier(value)
}

func (h *runtimeConfigurationHandlers) listRuntimeConfigs(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-configs", 2)
	if err != nil {
		handleError(w, err)
		return
	}
	afterName, afterVersion := "", ""
	if len(cursor) != 0 {
		afterName, afterVersion = cursor[0], cursor[1]
	}
	versions, err := h.RuntimeConfigs.ListVersions(r.Context(), afterName, afterVersion, pagelimit.FetchRows(limit))
	if err != nil {
		handleError(w, err)
		return
	}
	versions, page, err := paginate(h.cursors, versions, limit, "runtime-configs", func(last runtimeconfig.Version) []string {
		return []string{last.Ref.Name, last.Ref.Version}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	items := make([]runtimeConfigResourceResponse, len(versions))
	for index := range versions {
		items[index] = runtimeConfigResource(versions[index])
	}
	writeJSON(w, http.StatusOK, runtimeConfigPageResponse{Items: items, Page: page})
}

func (h *runtimeConfigurationHandlers) getRuntimeConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	name, version := r.PathValue("name"), r.PathValue("version")
	if !validRuntimeInfrastructureID(name) || !validRuntimeConfigVersion(version) {
		handleError(w, errInvalidRequest)
		return
	}
	resource, err := h.RuntimeConfigs.GetVersion(r.Context(), name, version)
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(resource.Ref.Digest))
	writeJSON(w, http.StatusOK, runtimeConfigResource(resource))
}

func (h *runtimeConfigurationHandlers) publishRuntimeConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	body, err := readBoundedRuntimeJSON(w, r, maximumRuntimeConfigBodyBytes)
	if err != nil {
		handleError(w, err)
		return
	}
	defer clear(body)
	result, err := h.RuntimeConfigs.Publish(
		r.Context(), body, idempotencyKey, principalUserID(r.Context()),
	)
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, result.Version.Ref.Name,
		); err != nil {
			handleError(w, err)
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

func (h *runtimeConfigurationHandlers) listRuntimeLabels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-labels", 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	bindings, err := h.RuntimeConfigs.ListBindings(r.Context(), after, pagelimit.FetchRows(limit))
	if err != nil {
		handleError(w, err)
		return
	}
	bindings, page, err := paginate(h.cursors, bindings, limit, "runtime-labels", func(last runtimeconfig.Binding) []string {
		return []string{last.Label}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	items := make([]runtimeLabelResponse, len(bindings))
	for index := range bindings {
		items[index] = runtimeLabelResource(bindings[index])
	}
	writeJSON(w, http.StatusOK, runtimeLabelPageResponse{Items: items, Page: page})
}

func (h *runtimeConfigurationHandlers) getRuntimeLabel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	label := r.PathValue("label")
	if !validRuntimeInfrastructureID(label) {
		handleError(w, errInvalidRequest)
		return
	}
	binding, err := h.RuntimeConfigs.GetBinding(r.Context(), label)
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(binding.Revision, 10)))
	writeJSON(w, http.StatusOK, runtimeLabelResource(binding))
}

func (h *runtimeConfigurationHandlers) putRuntimeLabel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	label := r.PathValue("label")
	if !validRuntimeInfrastructureID(label) {
		handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	mode, expectedRevision, err := runtimeLabelPutPrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var request runtimeLabelMutationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	actor := principalUserID(r.Context())
	var result runtimeconfig.BindingMutationResult
	if mode == labelMutationCreate {
		result, err = h.RuntimeConfigs.CreateBinding(
			r.Context(), label, request.Config, idempotencyKey, actor, h.Now(),
		)
	} else {
		result, err = h.RuntimeConfigs.Rebind(
			r.Context(), label, expectedRevision, request.Config, idempotencyKey, actor, h.Now(),
		)
	}
	if err != nil {
		handleError(w, err)
		return
	}
	if result.Binding == nil {
		handleError(w, errors.New("Runtime label mutation returned no binding"))
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, label,
		); err != nil {
			handleError(w, err)
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

func (h *runtimeConfigurationHandlers) deleteRuntimeLabel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	label := r.PathValue("label")
	if !validRuntimeInfrastructureID(label) {
		handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	expectedRevision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.RuntimeConfigs.DeleteBinding(
		r.Context(), label, expectedRevision, idempotencyKey,
		principalUserID(r.Context()), h.Now(),
	)
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Deleted {
		handleError(w, errors.New("Runtime label mutation did not delete the binding"))
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, label,
		); err != nil {
			handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_label.delete", label, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *runtimeConfigurationHandlers) listRuntimeCredentials(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-credentials", 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	items, err := h.RuntimeCredentials.List(r.Context(), after, pagelimit.FetchRows(limit))
	if err != nil {
		handleError(w, err)
		return
	}
	items, page, err := paginate(h.cursors, items, limit, "runtime-credentials", func(last credentials.RuntimeCredentialMetadata) []string {
		return []string{last.CredentialID}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, runtimeCredentialPageResponse{Items: items, Page: page})
}

func (h *runtimeConfigurationHandlers) getRuntimeCredential(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	id := r.PathValue("credentialId")
	if !validRuntimeCredentialID(id) {
		handleError(w, errInvalidRequest)
		return
	}
	metadata, err := h.RuntimeCredentials.Get(r.Context(), id)
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, metadata)
}

func (h *runtimeConfigurationHandlers) createRuntimeCredential(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var request createRuntimeCredentialRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	defer request.Material.Destroy()
	if !validRuntimeCredentialID(request.CredentialID) {
		handleError(w, errInvalidRequest)
		return
	}
	result, err := h.RuntimeCredentials.Create(r.Context(), credentials.RuntimeCredentialCreateRequest{
		CredentialID: request.CredentialID, Material: request.Material,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, "runtime:"+result.Credential.CredentialID,
		); err != nil {
			handleError(w, err)
			return
		}
	}
	h.auditRuntimeMutation(r, "runtime_credential.create", result.Credential.CredentialID, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Credential)
}

func (h *runtimeConfigurationHandlers) deleteRuntimeCredential(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	id := r.PathValue("credentialId")
	if !validRuntimeCredentialID(id) {
		handleError(w, errInvalidRequest)
		return
	}
	// The tombstone makes deletion idempotent by credential ID, so the key is
	// optional and not bound; a supplied one must still be well formed.
	if len(r.Header.Values(idempotencyKeyHeader)) != 0 {
		if _, err := requireIdempotencyKey(r); err != nil {
			handleError(w, err)
			return
		}
	}
	result, err := h.RuntimeCredentials.Delete(
		r.Context(), id, principalUserID(r.Context()),
	)
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, "runtime:"+id,
		); err != nil {
			handleError(w, err)
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

func readBoundedRuntimeJSON(w http.ResponseWriter, r *http.Request, maximum int64) ([]byte, error) {
	if maximum <= 0 || r.ContentLength > maximum {
		return nil, fmt.Errorf("%w: JSON body is too large", errRequestTooLarge)
	}
	body := http.MaxBytesReader(w, r.Body, maximum)
	data, err := io.ReadAll(body)
	if err != nil {
		if exceedsBodyLimit(err) {
			return nil, fmt.Errorf("%w: JSON body is too large", errRequestTooLarge)
		}
		return nil, errors.New("read RuntimeConfig request")
	}
	if len(data) == 0 {
		return nil, errInvalidRequest
	}
	return data, nil
}

func (h *runtimeConfigurationHandlers) auditRuntimeMutation(r *http.Request, action, resourceID string, replayed bool) {
	h.Logger.InfoContext(
		r.Context(), "Runtime Operations mutation",
		"audit_action", action,
		"actor_id", principalUserID(r.Context()),
		"request_id", requestid.From(r.Context()),
		"resource_id", resourceID,
		"replayed", replayed,
	)
}
