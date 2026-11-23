package public

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
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

func (h *runtimeConfigurationHandlers) listRuntimeConfigs(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-configs", 2)
	if err != nil {
		return err
	}
	afterName, afterVersion := "", ""
	if len(cursor) != 0 {
		afterName, afterVersion = cursor[0], cursor[1]
	}
	versions, err := h.RuntimeConfigs.ListVersions(r.Context(), afterName, afterVersion, pagelimit.FetchRows(limit))
	if err != nil {
		return err
	}
	versions, page, err := paginate(h.cursors, versions, limit, "runtime-configs", func(last runtimeconfig.Version) []string {
		return []string{last.Ref.Name, last.Ref.Version}
	})
	if err != nil {
		return err
	}
	items := make([]runtimeConfigResourceResponse, len(versions))
	for index := range versions {
		items[index] = runtimeConfigResource(versions[index])
	}
	writeJSON(w, http.StatusOK, runtimeConfigPageResponse{Items: items, Page: page})
	return nil
}

func (h *runtimeConfigurationHandlers) getRuntimeConfig(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	name, version := r.PathValue("name"), r.PathValue("version")
	if !validRuntimeInfrastructureID(name) || !validRuntimeConfigVersion(version) {
		return errInvalidRequest
	}
	resource, err := h.RuntimeConfigs.GetVersion(r.Context(), name, version)
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(resource.Ref.Digest))
	writeJSON(w, http.StatusOK, runtimeConfigResource(resource))
	return nil
}

func (h *runtimeConfigurationHandlers) publishRuntimeConfig(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	// The document is published as sent: its own strict decoder checks the
	// exact bytes, including duplicate members and nulls.
	body, err := readJSONBody(w, r, maximumRuntimeConfigBodyBytes)
	if err == nil && len(body) == 0 {
		err = fmt.Errorf("%w: RuntimeConfig document is required", errInvalidRequest)
	}
	if err != nil {
		return err
	}
	defer clear(body)
	result, err := h.RuntimeConfigs.Publish(
		r.Context(), body, idempotencyKey, principalUserID(r.Context()),
	)
	if err != nil {
		return err
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, result.Version.Ref.Name,
		); err != nil {
			return err
		}
	}
	h.auditRuntimeMutation(r, "runtime_config.publish", result.Version.Ref.Name, result.Replayed)
	w.Header().Set("ETag", strconv.Quote(result.Version.Ref.Digest))
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, runtimeConfigResource(result.Version))
	return nil
}

func (h *runtimeConfigurationHandlers) listRuntimeLabels(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-labels", 1)
	if err != nil {
		return err
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	bindings, err := h.RuntimeConfigs.ListBindings(r.Context(), after, pagelimit.FetchRows(limit))
	if err != nil {
		return err
	}
	bindings, page, err := paginate(h.cursors, bindings, limit, "runtime-labels", func(last runtimeconfig.Binding) []string {
		return []string{last.Label}
	})
	if err != nil {
		return err
	}
	items := make([]runtimeLabelResponse, len(bindings))
	for index := range bindings {
		items[index] = runtimeLabelResource(bindings[index])
	}
	writeJSON(w, http.StatusOK, runtimeLabelPageResponse{Items: items, Page: page})
	return nil
}

func (h *runtimeConfigurationHandlers) getRuntimeLabel(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	label := r.PathValue("label")
	if !validRuntimeInfrastructureID(label) {
		return errInvalidRequest
	}
	binding, err := h.RuntimeConfigs.GetBinding(r.Context(), label)
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(binding.Revision, 10)))
	writeJSON(w, http.StatusOK, runtimeLabelResource(binding))
	return nil
}

func (h *runtimeConfigurationHandlers) putRuntimeLabel(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	label := r.PathValue("label")
	if !validRuntimeInfrastructureID(label) {
		return errInvalidRequest
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	mode, expectedRevision, err := runtimeLabelPutPrecondition(r)
	if err != nil {
		return err
	}
	var request runtimeLabelMutationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
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
		return err
	}
	if result.Binding == nil {
		return errors.New("Runtime label mutation returned no binding")
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, label,
		); err != nil {
			return err
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
	return nil
}

func (h *runtimeConfigurationHandlers) deleteRuntimeLabel(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	label := r.PathValue("label")
	if !validRuntimeInfrastructureID(label) {
		return errInvalidRequest
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	expectedRevision, err := runtimeLabelDeletePrecondition(r)
	if err != nil {
		return err
	}
	result, err := h.RuntimeConfigs.DeleteBinding(
		r.Context(), label, expectedRevision, idempotencyKey,
		principalUserID(r.Context()), h.Now(),
	)
	if err != nil {
		return err
	}
	if !result.Deleted {
		return errors.New("Runtime label mutation did not delete the binding")
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, label,
		); err != nil {
			return err
		}
	}
	h.auditRuntimeMutation(r, "runtime_label.delete", label, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (h *runtimeConfigurationHandlers) listRuntimeCredentials(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "runtime-credentials", 1)
	if err != nil {
		return err
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	items, err := h.RuntimeCredentials.List(r.Context(), after, pagelimit.FetchRows(limit))
	if err != nil {
		return err
	}
	items, page, err := paginate(h.cursors, items, limit, "runtime-credentials", func(last credentials.RuntimeCredentialMetadata) []string {
		return []string{last.CredentialID}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, runtimeCredentialPageResponse{Items: items, Page: page})
	return nil
}

func (h *runtimeConfigurationHandlers) getRuntimeCredential(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	id := r.PathValue("credentialId")
	if !validRuntimeCredentialID(id) {
		return errInvalidRequest
	}
	metadata, err := h.RuntimeCredentials.Get(r.Context(), id)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, metadata)
	return nil
}

func (h *runtimeConfigurationHandlers) createRuntimeCredential(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	var request createRuntimeCredentialRequest
	if err := decodeOpaqueJSONBounded(w, r, &request, maxJSONRequestSize); err != nil {
		return err
	}
	defer request.Material.Destroy()
	if !validRuntimeCredentialID(request.CredentialID) {
		return errInvalidRequest
	}
	result, err := h.RuntimeCredentials.Create(r.Context(), credentials.RuntimeCredentialCreateRequest{
		CredentialID: request.CredentialID, Material: request.Material,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		return err
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, "runtime:"+result.Credential.CredentialID,
		); err != nil {
			return err
		}
	}
	h.auditRuntimeMutation(r, "runtime_credential.create", result.Credential.CredentialID, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Credential)
	return nil
}

func (h *runtimeConfigurationHandlers) deleteRuntimeCredential(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	id := r.PathValue("credentialId")
	if !validRuntimeCredentialID(id) {
		return errInvalidRequest
	}
	// The tombstone makes deletion idempotent by credential ID, so the key is
	// optional and not bound; a supplied one must still be well formed.
	if err := validateOptionalIdempotencyKey(r); err != nil {
		return err
	}
	result, err := h.RuntimeCredentials.Delete(
		r.Context(), id, principalUserID(r.Context()),
	)
	if err != nil {
		return err
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, "runtime:"+id,
		); err != nil {
			return err
		}
	}
	h.auditRuntimeMutation(r, "runtime_credential.delete", id, result.Replayed)
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
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
	return requiredRevision(r, "binding DELETE")
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
