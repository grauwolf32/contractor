package config

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
	"golang.org/x/sys/unix"
)

const (
	managedDirectoryMode = 0o750
	managedManifestMode  = 0o640
)

var configurationSubtrees = []string{
	"workflows",
	"agent-templates",
	"model-policies",
	"llm-gateways",
	"execution-configs",
	"instructions",
}

// ModelPolicyPublication is the typed, path-free body accepted by the
// managed configuration publisher. Pointer fields distinguish omission from
// an explicitly invalid zero value.
type ModelPolicyPublication struct {
	Model           string   `json:"model"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	MaxModelCalls   *int     `json:"maxModelCalls,omitempty"`
	MaxToolCalls    *int     `json:"maxToolCalls,omitempty"`
	MaxWorkerCalls  *int     `json:"maxWorkerCalls,omitempty"`
	MaxTotalTokens  *int     `json:"maxTotalTokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
}

type CredentialManagerPublication struct {
	Implementation string `json:"implementation"`
	ManagementURL  string `json:"managementUrl"`
}

type LLMGatewayPublication struct {
	Protocol          string                        `json:"protocol"`
	URL               string                        `json:"url"`
	CredentialManager *CredentialManagerPublication `json:"credentialManager,omitempty"`
}

type PublicationRequest struct {
	Kind           ConfigurationKind
	Name           string
	Version        string
	ModelPolicy    *ModelPolicyPublication
	LLMGateway     *LLMGatewayPublication
	IdempotencyKey string
	ActorID        string
}

type PublicationResult struct {
	Resource ConfigurationResource
	Replayed bool
}

// PublicationAudit contains only safe metadata. In particular, it never
// contains a Gateway credential, instruction text, or a raw idempotency key.
type PublicationAudit struct {
	Kind                 ConfigurationKind
	Name                 string
	Version              string
	Digest               string
	RequestDigest        string
	IdempotencyKeyDigest string
	ActorID              string
	PublishedAt          time.Time
}

type PublicationAuditRecorder interface {
	RecordConfigurationPublication(context.Context, PublicationAudit) error
}

type ManagerOptions struct {
	OperatorRoot string
	ManagedRoot  string
	Descriptors  Descriptors
	Audit        PublicationAuditRecorder
	Logger       *slog.Logger
	Now          func() time.Time

	// AfterDurablePublish is a test seam for the rename-before-snapshot-swap
	// crash window. Production callers leave it nil.
	AfterDurablePublish func(ConfigurationResource) error
}

type idempotentPublication struct {
	requestDigest string
}

// Manager owns the process-local current immutable configuration snapshot and
// serializes managed publications. Filesystem locking across Server processes
// is deliberately outside the single-Server first slice.
type Manager struct {
	publicationMu sync.Mutex
	current       atomic.Pointer[Snapshot]

	operatorRoot string
	managedRoot  string
	descriptors  Descriptors
	audit        PublicationAuditRecorder
	logger       *slog.Logger
	now          func() time.Time
	afterPublish func(ConfigurationResource) error
	idempotency  map[string]idempotentPublication
}

func NewManager(options ManagerOptions) (*Manager, error) {
	operatorRoot, err := requireStrictRoot(options.OperatorRoot, false)
	if err != nil {
		return nil, fmt.Errorf("operator configuration root: %w", err)
	}
	managedRoot, err := requireStrictRoot(options.ManagedRoot, true)
	if err != nil {
		return nil, fmt.Errorf("managed configuration root: %w", err)
	}
	if rootsOverlap(operatorRoot, managedRoot) {
		return nil, fmt.Errorf("operator and managed configuration roots must not overlap")
	}
	snapshot, err := LoadUnion(operatorRoot, managedRoot, options.Descriptors)
	if err != nil {
		return nil, err
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	manager := &Manager{
		operatorRoot: operatorRoot,
		managedRoot:  managedRoot,
		descriptors:  options.Descriptors,
		audit:        options.Audit,
		logger:       logger,
		now:          now,
		afterPublish: options.AfterDurablePublish,
		idempotency:  make(map[string]idempotentPublication),
	}
	manager.current.Store(snapshot)
	return manager, nil
}

func (m *Manager) Snapshot() *Snapshot { return m.current.Load() }

func (m *Manager) Counts() Counts { return m.Snapshot().Counts() }

func (m *Manager) Workflow(raw string) (ResolvedWorkflow, error) {
	return m.Snapshot().Workflow(raw)
}

func (m *Manager) Workflows() []ResolvedWorkflow { return m.Snapshot().Workflows() }

func (m *Manager) ResolveRunWorkflow(
	ctx context.Context,
	raw string,
	patch ExecutionConfigPatch,
	credentials CredentialLookup,
) (ResolvedWorkflow, error) {
	return m.Snapshot().ResolveRunWorkflow(ctx, raw, patch, credentials)
}

func (m *Manager) Configurations(kind ConfigurationKind) ([]ConfigurationResource, error) {
	return m.Snapshot().Configurations(kind)
}

func (m *Manager) Configuration(
	kind ConfigurationKind, raw string,
) (ConfigurationResource, error) {
	return m.Snapshot().Configuration(kind, raw)
}

func (m *Manager) Publish(
	ctx context.Context, request PublicationRequest,
) (PublicationResult, error) {
	candidate, err := preparePublication(request)
	if err != nil {
		return PublicationResult{}, err
	}

	m.publicationMu.Lock()
	defer m.publicationMu.Unlock()

	loaded, err := LoadUnion(m.operatorRoot, m.managedRoot, m.descriptors)
	if err != nil {
		return PublicationResult{}, fmt.Errorf("reload configuration roots before publication: %w", err)
	}
	if previous, exists := m.idempotency[request.IdempotencyKey]; exists {
		if previous.requestDigest != candidate.requestDigest {
			return PublicationResult{}, fmt.Errorf("%w: Idempotency-Key was already used for another publication", ErrPublicationConflict)
		}
		existing, existingErr := loaded.Configuration(request.Kind, candidate.selector.String())
		if existingErr != nil || existing.Source != ConfigurationSourceManaged ||
			existing.Ref.Digest != candidate.resource.Ref.Digest ||
			!snapshotsEqual(loaded, m.current.Load()) {
			return PublicationResult{}, fmt.Errorf("configuration roots changed after the original publication")
		}
		return PublicationResult{Resource: existing, Replayed: true}, nil
	}
	existing, existingErr := loaded.Configuration(request.Kind, candidate.selector.String())
	if existingErr == nil {
		if existing.Source != ConfigurationSourceManaged || existing.Ref.Digest != candidate.resource.Ref.Digest {
			return PublicationResult{}, fmt.Errorf("%w: %s already exists", ErrPublicationConflict, candidate.selector)
		}
		expected := m.current.Load().withPublication(candidate)
		if !snapshotsEqual(loaded, expected) {
			return PublicationResult{}, fmt.Errorf("configuration roots changed outside the publication manager")
		}
		// This includes recovery from a crash after durable rename and before
		// the previous process could swap its in-memory snapshot.
		m.current.Store(loaded)
		m.rememberPublication(request.IdempotencyKey, candidate.requestDigest)
		m.recordAudit(ctx, request, candidate)
		return PublicationResult{Resource: existing, Replayed: true}, nil
	}
	if !errors.Is(existingErr, ErrConfigurationNotFound) {
		return PublicationResult{}, existingErr
	}
	if !snapshotsEqual(loaded, m.current.Load()) {
		return PublicationResult{}, fmt.Errorf("configuration roots changed outside the publication manager")
	}

	next := loaded.withPublication(candidate)
	if err := m.writeDurableManifest(candidate); err != nil {
		return PublicationResult{}, err
	}
	if m.afterPublish != nil {
		if err := m.afterPublish(candidate.resource); err != nil {
			return PublicationResult{}, err
		}
	}
	m.current.Store(next)
	m.rememberPublication(request.IdempotencyKey, candidate.requestDigest)
	m.recordAudit(ctx, request, candidate)
	return PublicationResult{Resource: candidate.resource}, nil
}

type publicationCandidate struct {
	selector      Selector
	resource      ConfigurationResource
	policy        *contracts.ResolvedModelPolicy
	gateway       *contracts.ResolvedLLMGatewayConfig
	canonicalYAML []byte
	requestDigest string
	subtree       string
}

func preparePublication(request PublicationRequest) (publicationCandidate, error) {
	if err := validatePublicationKey(request.IdempotencyKey); err != nil {
		return publicationCandidate{}, err
	}
	if len(request.Name) > 128 || len(request.Version) > 64 {
		return publicationCandidate{}, fmt.Errorf("%w: name or version exceeds the public contract", ErrInvalidPublication)
	}
	selector, err := validateMetadata(&metadataSource{Name: request.Name, Version: request.Version})
	if err != nil {
		return publicationCandidate{}, fmt.Errorf("%w: %v", ErrInvalidPublication, err)
	}

	switch request.Kind {
	case ConfigurationModelPolicies:
		if request.ModelPolicy == nil || request.LLMGateway != nil {
			return publicationCandidate{}, fmt.Errorf("%w: model-policies requires exactly one modelPolicy body", ErrInvalidPublication)
		}
		if utf8.RuneCountInString(request.ModelPolicy.Model) > 256 {
			return publicationCandidate{}, fmt.Errorf("%w: model exceeds 256 characters", ErrInvalidPublication)
		}
		spec := modelPolicySpecSource{
			Model:           request.ModelPolicy.Model,
			MaxOutputTokens: cloneInt(request.ModelPolicy.MaxOutputTokens),
			MaxModelCalls:   cloneInt(request.ModelPolicy.MaxModelCalls),
			MaxToolCalls:    cloneInt(request.ModelPolicy.MaxToolCalls),
			MaxWorkerCalls:  cloneInt(request.ModelPolicy.MaxWorkerCalls),
			MaxTotalTokens:  cloneInt(request.ModelPolicy.MaxTotalTokens),
			Temperature:     cloneFloat(request.ModelPolicy.Temperature),
		}
		if err := validateModelPolicySpec(&spec); err != nil {
			return publicationCandidate{}, fmt.Errorf("%w: %v", ErrInvalidPublication, err)
		}
		policy := contracts.ResolvedModelPolicy{
			Ref:   contracts.ModelPolicyRef{PolicyID: selector.ID, Version: selector.Version},
			Model: spec.Model, MaxOutputTokens: optionalIntValue(spec.MaxOutputTokens),
			MaxModelCalls: optionalIntValue(spec.MaxModelCalls), MaxToolCalls: optionalIntValue(spec.MaxToolCalls),
			MaxWorkerCalls: optionalIntValue(spec.MaxWorkerCalls), MaxTotalTokens: optionalIntValue(spec.MaxTotalTokens),
			Temperature: cloneFloat(spec.Temperature),
		}
		digest, digestErr := modelPolicyDigest(selector, policy)
		if digestErr != nil {
			return publicationCandidate{}, fmt.Errorf("%w: %v", ErrInvalidPublication, digestErr)
		}
		policy.Ref.Digest = digest
		if err := policy.Validate(); err != nil {
			return publicationCandidate{}, fmt.Errorf("%w: %v", ErrInvalidPublication, err)
		}
		document := modelPolicyDocument{
			APIVersion: contracts.APIVersion, Kind: modelPolicyKind,
			Metadata: &metadataSource{Name: selector.ID, Version: selector.Version}, Spec: &spec,
		}
		encoded, encodeErr := yaml.Marshal(document)
		if encodeErr != nil {
			return publicationCandidate{}, fmt.Errorf("encode ModelPolicy YAML: %w", encodeErr)
		}
		resource := ConfigurationResource{
			Ref:  ConfigurationRef{Kind: request.Kind, Name: selector.ID, Version: selector.Version, Digest: digest},
			Body: modelPolicyResourceBody(policy), Source: ConfigurationSourceManaged,
		}
		return publicationCandidate{
			selector: selector, resource: resource, policy: &policy, canonicalYAML: encoded,
			requestDigest: digest, subtree: "model-policies",
		}, nil

	case ConfigurationLLMGateways:
		if request.LLMGateway == nil || request.ModelPolicy != nil {
			return publicationCandidate{}, fmt.Errorf("%w: llm-gateways requires exactly one llmGateway body", ErrInvalidPublication)
		}
		if len(request.LLMGateway.URL) > 2048 || request.LLMGateway.CredentialManager != nil &&
			len(request.LLMGateway.CredentialManager.ManagementURL) > 2048 {
			return publicationCandidate{}, fmt.Errorf("%w: Gateway URL exceeds 2048 bytes", ErrInvalidPublication)
		}
		spec := llmGatewayConfigSpecSource{
			Protocol: request.LLMGateway.Protocol, URL: request.LLMGateway.URL,
		}
		if request.LLMGateway.CredentialManager != nil {
			spec.CredentialManager = &llmCredentialManagerSpecSource{
				Implementation: request.LLMGateway.CredentialManager.Implementation,
				ManagementURL:  request.LLMGateway.CredentialManager.ManagementURL,
			}
		}
		gateway, resolveErr := resolveLLMGatewayConfig(selector, &spec)
		if resolveErr != nil {
			return publicationCandidate{}, fmt.Errorf("%w: %v", ErrInvalidPublication, resolveErr)
		}
		// Persist the normalized URLs used by the digest and in-memory value.
		spec.URL = gateway.URL
		if gateway.CredentialManager != nil {
			spec.CredentialManager.ManagementURL = gateway.CredentialManager.ManagementURL
		}
		document := llmGatewayConfigDocument{
			APIVersion: contracts.APIVersion, Kind: llmGatewayConfigKind,
			Metadata: &metadataSource{Name: selector.ID, Version: selector.Version}, Spec: &spec,
		}
		encoded, encodeErr := yaml.Marshal(document)
		if encodeErr != nil {
			return publicationCandidate{}, fmt.Errorf("encode LLMGatewayConfig YAML: %w", encodeErr)
		}
		resource := ConfigurationResource{
			Ref:  ConfigurationRef{Kind: request.Kind, Name: selector.ID, Version: selector.Version, Digest: gateway.Ref.Digest},
			Body: llmGatewayResourceBody(gateway), Source: ConfigurationSourceManaged,
		}
		return publicationCandidate{
			selector: selector, resource: resource, gateway: &gateway, canonicalYAML: encoded,
			requestDigest: gateway.Ref.Digest, subtree: "llm-gateways",
		}, nil

	case ConfigurationAgentTemplates, ConfigurationExecutionConfigs:
		return publicationCandidate{}, fmt.Errorf("%w: %s is read-only", ErrInvalidPublication, request.Kind)
	default:
		return publicationCandidate{}, ErrInvalidConfigurationKind
	}
}

func validatePublicationKey(value string) error {
	if len(value) == 0 || len(value) > 128 {
		return fmt.Errorf("%w: Idempotency-Key must contain 1 to 128 characters", ErrInvalidPublication)
	}
	for index, character := range value {
		valid := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || index > 0 && strings.ContainsRune("._:-", character)
		if !valid {
			return fmt.Errorf("%w: Idempotency-Key contains an invalid character", ErrInvalidPublication)
		}
	}
	return nil
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func (s *Snapshot) withPublication(candidate publicationCandidate) *Snapshot {
	policies := s.policies
	gateways := s.gateways
	if candidate.policy != nil {
		policies = cloneMap(s.policies)
		policies[candidate.selector.String()] = cloneModelPolicy(*candidate.policy)
	}
	if candidate.gateway != nil {
		gateways = cloneMap(s.gateways)
		gateways[candidate.selector.String()] = cloneLLMGatewayConfig(*candidate.gateway)
	}
	sources := cloneMap(s.sources)
	sources[configurationSourceKey(candidate.resource.Ref.Kind, candidate.selector.String())] = ConfigurationSourceManaged
	return newSnapshot(
		s.workflows, s.templates, policies, gateways, s.executionConfigs, s.instructions, sources,
	)
}

func cloneMap[K comparable, V any](source map[K]V) map[K]V {
	result := make(map[K]V, len(source)+1)
	for key, value := range source {
		result[key] = value
	}
	return result
}

func snapshotsEqual(left, right *Snapshot) bool {
	return left != nil && right != nil && reflect.DeepEqual(left, right)
}

func (m *Manager) writeDurableManifest(candidate publicationCandidate) error {
	directoryPath := filepath.Join(m.managedRoot, candidate.subtree)
	directoryFD, err := unix.Open(directoryPath, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return fmt.Errorf("open managed configuration subtree: %w", err)
	}
	defer unix.Close(directoryFD)

	random := make([]byte, 16)
	if _, err := cryptorand.Read(random); err != nil {
		return fmt.Errorf("generate publication temporary name: %w", err)
	}
	temporaryName := ".contractor-publish-" + hex.EncodeToString(random) + ".tmp"
	finalName := candidate.selector.ID + "@" + candidate.selector.Version + ".yaml"
	temporaryFD, err := unix.Openat(
		directoryFD,
		temporaryName,
		unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC,
		managedManifestMode,
	)
	if err != nil {
		return fmt.Errorf("create publication temporary file: %w", err)
	}
	temporary := os.NewFile(uintptr(temporaryFD), temporaryName)
	cleanup := true
	defer func() {
		if temporary != nil {
			_ = temporary.Close()
		}
		if cleanup {
			_ = unix.Unlinkat(directoryFD, temporaryName, 0)
		}
	}()
	if _, err := temporary.Write(candidate.canonicalYAML); err != nil {
		return fmt.Errorf("write publication temporary file: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		return fmt.Errorf("flush publication temporary file: %w", err)
	}
	if err := temporary.Close(); err != nil {
		temporary = nil
		return fmt.Errorf("close publication temporary file: %w", err)
	}
	temporary = nil
	if err := unix.Renameat2(directoryFD, temporaryName, directoryFD, finalName, unix.RENAME_NOREPLACE); err != nil {
		if errors.Is(err, unix.EEXIST) {
			return fmt.Errorf("%w: %s already exists", ErrPublicationConflict, candidate.selector)
		}
		return fmt.Errorf("atomically publish managed manifest: %w", err)
	}
	cleanup = false
	if err := unix.Fsync(directoryFD); err != nil {
		return fmt.Errorf("flush managed configuration directory: %w", err)
	}
	return nil
}

func (m *Manager) rememberPublication(key, requestDigest string) {
	m.idempotency[key] = idempotentPublication{requestDigest: requestDigest}
}

func (m *Manager) recordAudit(
	ctx context.Context, request PublicationRequest, candidate publicationCandidate,
) {
	if m.audit == nil {
		return
	}
	keyDigest := digestBytes([]byte(request.IdempotencyKey))
	err := m.audit.RecordConfigurationPublication(ctx, PublicationAudit{
		Kind: request.Kind, Name: candidate.selector.ID, Version: candidate.selector.Version,
		Digest: candidate.resource.Ref.Digest, RequestDigest: candidate.requestDigest,
		IdempotencyKeyDigest: keyDigest, ActorID: request.ActorID, PublishedAt: m.now().UTC(),
	})
	if err != nil {
		m.logger.Error("record non-authoritative configuration publication audit", "error", err)
	}
}

func requireStrictRoot(path string, create bool) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("path is required")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	info, err := os.Lstat(absolute)
	if errors.Is(err, os.ErrNotExist) && create {
		if err := os.MkdirAll(absolute, managedDirectoryMode); err != nil {
			return "", err
		}
		info, err = os.Lstat(absolute)
	}
	if err != nil {
		return "", err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", fmt.Errorf("root must be a real directory, not a symlink")
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", err
	}
	absolute = resolved
	for _, subtree := range configurationSubtrees {
		child := filepath.Join(absolute, subtree)
		childInfo, childErr := os.Lstat(child)
		if errors.Is(childErr, os.ErrNotExist) && create {
			childErr = os.Mkdir(child, managedDirectoryMode)
			if childErr == nil {
				childInfo, childErr = os.Lstat(child)
			}
		}
		if childErr != nil {
			return "", fmt.Errorf("configuration subtree %s: %w", subtree, childErr)
		}
		if childInfo.Mode()&os.ModeSymlink != 0 || !childInfo.IsDir() {
			return "", fmt.Errorf("configuration subtree %s must be a real directory", subtree)
		}
	}
	return absolute, nil
}

func rootsOverlap(left, right string) bool {
	leftRelative, leftErr := filepath.Rel(left, right)
	rightRelative, rightErr := filepath.Rel(right, left)
	return leftErr == nil && (leftRelative == "." || leftRelative != ".." && !strings.HasPrefix(leftRelative, ".."+string(filepath.Separator))) ||
		rightErr == nil && (rightRelative == "." || rightRelative != ".." && !strings.HasPrefix(rightRelative, ".."+string(filepath.Separator)))
}
