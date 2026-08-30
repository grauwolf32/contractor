package litellm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

const (
	defaultConnectTimeout = 3 * time.Second
	defaultRequestTimeout = 15 * time.Second
	defaultResponseBytes  = 64 * 1024
	maximumResponseBytes  = 1 << 20
	maximumRequestBytes   = 64 * 1024
	liteLLMKeyType        = "llm_api"
	liteLLMAllowedRoute   = "llm_api_routes"
	managerImplementation = contracts.LiteLLMVirtualKeysManager
	managerUserAgent      = "contractor/litellm-virtual-keys@1"
	confirmedNotFoundBody = "{'error': 'No keys found'}"
)

var (
	operationIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`)
	tokenIDPattern     = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

type ModelPolicyLookup interface {
	ModelPolicy(string) (contracts.ResolvedModelPolicy, error)
}

type Options struct {
	ConnectTimeout   time.Duration
	RequestTimeout   time.Duration
	MaxResponseBytes int64
}

// Manager implements credentials.GatewayCredentialManager for the pinned
// LiteLLM virtual-key API. It has no logger and returns only stable Contractor
// errors so provider bodies and Authorization values cannot escape.
type Manager struct {
	bindings         *AdminBindings
	policies         ModelPolicyLookup
	client           *http.Client
	maxResponseBytes int64
}

func (m *Manager) String() string   { return "litellm.Manager([REDACTED])" }
func (m *Manager) GoString() string { return "litellm.Manager([REDACTED])" }

func NewManager(bindings *AdminBindings, policies ModelPolicyLookup, options Options) (*Manager, error) {
	if bindings == nil || policies == nil {
		return nil, fmt.Errorf("%w: LiteLLM manager dependencies are incomplete", credentials.ErrManagerUnavailable)
	}
	if options.ConnectTimeout == 0 {
		options.ConnectTimeout = defaultConnectTimeout
	}
	if options.RequestTimeout == 0 {
		options.RequestTimeout = defaultRequestTimeout
	}
	if options.MaxResponseBytes == 0 {
		options.MaxResponseBytes = defaultResponseBytes
	}
	if options.ConnectTimeout <= 0 || options.ConnectTimeout > time.Minute ||
		options.RequestTimeout <= 0 || options.RequestTimeout > 2*time.Minute ||
		options.MaxResponseBytes < 1024 || options.MaxResponseBytes > maximumResponseBytes {
		return nil, fmt.Errorf("%w: LiteLLM manager bounds are invalid", credentials.ErrManagerUnavailable)
	}
	dialer := &net.Dialer{Timeout: options.ConnectTimeout, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy:                 nil,
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          16,
		MaxIdleConnsPerHost:   4,
		MaxConnsPerHost:       8,
		IdleConnTimeout:       30 * time.Second,
		TLSHandshakeTimeout:   options.ConnectTimeout,
		ResponseHeaderTimeout: options.RequestTimeout,
		ExpectContinueTimeout: time.Second,
		TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12},
	}
	client := &http.Client{
		Transport: transport,
		Timeout:   options.RequestTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	return &Manager{
		bindings: bindings, policies: policies, client: client,
		maxResponseBytes: options.MaxResponseBytes,
	}, nil
}

func (m *Manager) ValidateCreate(ctx context.Context, request credentials.ManagerCreateRequest) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, _, err := m.createPayload(ctx, request)
	return err
}

func (m *Manager) Create(
	ctx context.Context,
	request credentials.ManagerCreateRequest,
) (credentials.GeneratedCredential, error) {
	payload, binding, err := m.createPayload(ctx, request)
	if err != nil {
		return credentials.GeneratedCredential{}, err
	}
	encoded, err := json.Marshal(payload)
	if err != nil || len(encoded) == 0 || len(encoded) > maximumRequestBytes {
		return credentials.GeneratedCredential{}, credentials.ErrGatewayUnavailable
	}
	status, response, err := m.post(ctx, binding, "/key/generate", encoded)
	if err != nil {
		return credentials.GeneratedCredential{}, err
	}
	defer wipe(response)
	if status != http.StatusOK {
		return credentials.GeneratedCredential{}, credentials.ErrGatewayUnavailable
	}
	generated, err := decodeGenerateResponse(response, payload)
	if err != nil {
		return credentials.GeneratedCredential{}, credentials.ErrGatewayUnavailable
	}
	return generated, nil
}

func (m *Manager) Delete(ctx context.Context, request credentials.ManagerDeleteRequest) error {
	binding, err := m.validateBase(
		ctx, request.OperationID, request.CredentialID, request.LLMGateway,
	)
	if err != nil {
		return err
	}
	if !tokenIDPattern.MatchString(request.RemoteKeyID) {
		return fmt.Errorf("%w: LiteLLM remote key ID is invalid", credentials.ErrInvalid)
	}
	return m.delete(ctx, binding, deleteKeyRequest{Keys: []string{request.RemoteKeyID}}, request.RemoteKeyID)
}

// RecoverCreate removes the deterministic alias before lifecycle retries key
// generation. It deliberately does not resolve ModelPolicies: cleanup remains
// possible even if an exact policy file was removed after a process crash.
func (m *Manager) RecoverCreate(ctx context.Context, request credentials.ManagerCreateRequest) error {
	binding, err := m.validateBase(
		ctx, request.OperationID, request.CredentialID, request.LLMGateway,
	)
	if err != nil {
		return err
	}
	alias, err := KeyAlias(request.LLMGateway.Ref, request.CredentialID)
	if err != nil {
		return err
	}
	return m.delete(ctx, binding, deleteKeyRequest{KeyAliases: []string{alias}}, alias)
}

func (m *Manager) createPayload(
	ctx context.Context,
	request credentials.ManagerCreateRequest,
) (generateKeyRequest, exactBinding, error) {
	binding, err := m.validateBase(
		ctx, request.OperationID, request.CredentialID, request.LLMGateway,
	)
	if err != nil {
		return generateKeyRequest{}, exactBinding{}, err
	}
	if request.Label != "" && (strings.TrimSpace(request.Label) == "" || !utf8.ValidString(request.Label) ||
		utf8.RuneCountInString(request.Label) > credentials.MaximumCredentialLabel) {
		return generateKeyRequest{}, exactBinding{}, fmt.Errorf("%w: credential label is invalid", credentials.ErrInvalid)
	}
	if err := request.Policy.Validate(); err != nil {
		return generateKeyRequest{}, exactBinding{}, err
	}
	refs := append([]contracts.ModelPolicyRef(nil), request.Policy.ModelPolicies...)
	sort.Slice(refs, func(left, right int) bool { return modelPolicyKey(refs[left]) < modelPolicyKey(refs[right]) })
	models := make([]string, 0, len(refs))
	seenModels := make(map[string]struct{}, len(refs))
	for _, ref := range refs {
		resolved, resolveErr := m.policies.ModelPolicy(ref.PolicyID + "@" + ref.Version)
		if resolveErr != nil || resolved.Ref != ref || resolved.Validate() != nil {
			return generateKeyRequest{}, exactBinding{}, fmt.Errorf("%w: exact ModelPolicy is unavailable", credentials.ErrInvalid)
		}
		if _, duplicate := seenModels[resolved.Model]; duplicate {
			continue
		}
		seenModels[resolved.Model] = struct{}{}
		models = append(models, resolved.Model)
	}
	sort.Strings(models)
	if len(models) == 0 || len(models) > 128 {
		return generateKeyRequest{}, exactBinding{}, fmt.Errorf("%w: derived LiteLLM models are invalid", credentials.ErrInvalid)
	}
	requestedPolicy := credentials.EffectiveGatewayPolicy{
		ModelPolicies:       append([]contracts.ModelPolicyRef(nil), refs...),
		Models:              append([]string(nil), models...),
		MaxBudget:           cloneFloat(request.Policy.MaxBudget),
		BudgetDuration:      request.Policy.BudgetDuration,
		TPMLimit:            cloneInt(request.Policy.TPMLimit),
		RPMLimit:            cloneInt(request.Policy.RPMLimit),
		MaxParallelRequests: cloneInt(request.Policy.MaxParallelRequests),
	}
	if err := requestedPolicy.Validate(); err != nil {
		return generateKeyRequest{}, exactBinding{}, err
	}
	alias, err := KeyAlias(request.LLMGateway.Ref, request.CredentialID)
	if err != nil {
		return generateKeyRequest{}, exactBinding{}, err
	}
	payload := generateKeyRequest{
		KeyAlias: alias,
		KeyType:  liteLLMKeyType,
		Models:   models,
		Metadata: keyMetadata{
			CredentialID:   request.CredentialID,
			GatewayID:      request.LLMGateway.Ref.GatewayID,
			GatewayVersion: request.LLMGateway.Ref.Version,
			GatewayDigest:  request.LLMGateway.Ref.Digest,
			OperationID:    request.OperationID,
			Label:          request.Label,
		},
		MaxBudget:           cloneFloat(request.Policy.MaxBudget),
		BudgetDuration:      request.Policy.BudgetDuration,
		TPMLimit:            cloneInt(request.Policy.TPMLimit),
		RPMLimit:            cloneInt(request.Policy.RPMLimit),
		MaxParallelRequests: cloneInt(request.Policy.MaxParallelRequests),
		ModelPolicies:       refs,
	}
	if encoded, err := json.Marshal(payload); err != nil || len(encoded) == 0 || len(encoded) > maximumRequestBytes {
		return generateKeyRequest{}, exactBinding{}, fmt.Errorf("%w: LiteLLM create payload exceeds its bound", credentials.ErrInvalid)
	}
	return payload, binding, nil
}

func (m *Manager) validateBase(
	ctx context.Context,
	operationID, credentialID string,
	gateway contracts.ResolvedLLMGatewayConfig,
) (exactBinding, error) {
	if err := ctx.Err(); err != nil {
		return exactBinding{}, err
	}
	if !operationIDPattern.MatchString(operationID) ||
		(contracts.LLMCredentialRef{CredentialID: credentialID}).Validate() != nil ||
		gateway.Validate() != nil || gateway.CredentialManager == nil ||
		gateway.CredentialManager.Implementation != managerImplementation {
		return exactBinding{}, fmt.Errorf("%w: LiteLLM manager request is invalid", credentials.ErrInvalid)
	}
	binding, err := m.bindings.bindingFor(gateway)
	if err != nil {
		return exactBinding{}, credentials.ErrManagerUnavailable
	}
	return binding, nil
}

// KeyAlias returns the bounded deterministic LiteLLM alias committed by the
// specification. It contains only a fixed prefix and a SHA-256 digest.
func KeyAlias(ref contracts.LLMGatewayConfigRef, credentialID string) (string, error) {
	if ref.ValidateRef() != nil || (contracts.LLMCredentialRef{CredentialID: credentialID}).Validate() != nil {
		return "", fmt.Errorf("%w: cannot derive LiteLLM key alias", credentials.ErrInvalid)
	}
	digest := sha256.New()
	_, _ = io.WriteString(digest, ref.Digest)
	_, _ = digest.Write([]byte{0})
	_, _ = io.WriteString(digest, credentialID)
	return "contractor-" + hex.EncodeToString(digest.Sum(nil)), nil
}

func (m *Manager) delete(
	ctx context.Context,
	binding exactBinding,
	payload deleteKeyRequest,
	expected string,
) error {
	encoded, err := json.Marshal(payload)
	if err != nil || len(encoded) == 0 || len(encoded) > maximumRequestBytes {
		return credentials.ErrGatewayUnavailable
	}
	status, response, err := m.post(ctx, binding, "/key/delete", encoded)
	if err != nil {
		return err
	}
	defer wipe(response)
	switch status {
	case http.StatusOK:
		var decoded deleteKeyResponse
		if decodeStrictJSON(response, &decoded) != nil || len(decoded.DeletedKeys) != 1 ||
			decoded.DeletedKeys[0] != expected {
			return credentials.ErrGatewayUnavailable
		}
		return nil
	case http.StatusNotFound:
		if confirmedKeyNotFound(response) {
			return nil
		}
		return credentials.ErrGatewayUnavailable
	default:
		return credentials.ErrGatewayUnavailable
	}
}

func (m *Manager) post(
	ctx context.Context,
	binding exactBinding,
	path string,
	payload []byte,
) (int, []byte, error) {
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, binding.managementURL+path, bytes.NewReader(payload),
	)
	if err != nil {
		return 0, nil, credentials.ErrGatewayUnavailable
	}
	request.Header.Set("Authorization", "Bearer "+binding.key.reveal())
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", managerUserAgent)
	response, err := m.client.Do(request)
	if err != nil {
		if contextErr := ctx.Err(); contextErr != nil {
			return 0, nil, contextErr
		}
		return 0, nil, credentials.ErrGatewayUnavailable
	}
	defer response.Body.Close()
	if !isStrictJSONContentType(response.Header.Values("Content-Type")) ||
		response.ContentLength > m.maxResponseBytes {
		return 0, nil, credentials.ErrGatewayUnavailable
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, m.maxResponseBytes+1))
	if err != nil || int64(len(body)) > m.maxResponseBytes {
		return 0, nil, credentials.ErrGatewayUnavailable
	}
	return response.StatusCode, body, nil
}

func decodeGenerateResponse(
	data []byte,
	request generateKeyRequest,
) (credentials.GeneratedCredential, error) {
	var response generateKeyResponse
	if decodeStrictJSON(data, &response) != nil || response.KeyAlias != request.KeyAlias ||
		!tokenIDPattern.MatchString(response.TokenID) || response.Token != response.TokenID ||
		!strings.HasPrefix(response.Key, "sk-") || !equalStrings(response.Models, request.Models) ||
		len(response.AllowedRoutes) != 1 || response.AllowedRoutes[0] != liteLLMAllowedRoute ||
		(response.Blocked != nil && *response.Blocked) || !metadataEqual(response.Metadata, request.Metadata) {
		return credentials.GeneratedCredential{}, errors.New("invalid LiteLLM key response")
	}
	token, err := credentials.NewToken(response.Key)
	if err != nil {
		return credentials.GeneratedCredential{}, errors.New("invalid LiteLLM generated key")
	}
	effective := credentials.EffectiveGatewayPolicy{
		ModelPolicies:       append([]contracts.ModelPolicyRef(nil), request.ModelPolicies...),
		Models:              append([]string(nil), request.Models...),
		MaxBudget:           cloneFloat(response.MaxBudget),
		TPMLimit:            cloneInt(response.TPMLimit),
		RPMLimit:            cloneInt(response.RPMLimit),
		MaxParallelRequests: cloneInt(response.MaxParallelRequests),
	}
	if response.BudgetDuration != nil {
		effective.BudgetDuration = *response.BudgetDuration
	}
	if err := effective.Validate(); err != nil {
		return credentials.GeneratedCredential{}, errors.New("invalid LiteLLM effective policy")
	}
	return credentials.GeneratedCredential{
		Token: token, RemoteKeyID: response.TokenID, EffectivePolicy: effective,
	}, nil
}

func confirmedKeyNotFound(data []byte) bool {
	var response liteLLMErrorResponse
	return decodeStrictJSON(data, &response) == nil && response.Error.Code == "404" &&
		response.Error.Type == "internal_server_error" && response.Error.Param == "None" &&
		response.Error.Message == confirmedNotFoundBody
}

func decodeStrictJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return errors.New("trailing JSON value")
	}
	return nil
}

func isStrictJSONContentType(values []string) bool {
	if len(values) != 1 {
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	return err == nil && mediaType == "application/json" && len(parameters) == 0
}

func metadataEqual(left, right keyMetadata) bool {
	return left == right
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func modelPolicyKey(ref contracts.ModelPolicyRef) string {
	return ref.PolicyID + "\x00" + ref.Version + "\x00" + ref.Digest
}

func cloneFloat(value *float64) *float64 {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

var _ credentials.GatewayCredentialManager = (*Manager)(nil)
