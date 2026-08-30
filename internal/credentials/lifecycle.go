package credentials

import (
	"context"
	cryptorand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const maximumCredentialRunReferences = 128

var (
	ErrGatewayUnavailable = errors.New("Gateway credential operation is unavailable")
	ErrRecoveryRequired   = errors.New("credential operation recovery is required")
)

type CredentialInUseError struct {
	RunIDs []string
}

func (e *CredentialInUseError) Error() string {
	return "LLM credential is pinned by a non-terminal WorkflowRun"
}

type GatewayLookup interface {
	LLMGateway(string) (contracts.ResolvedLLMGatewayConfig, error)
}

type NonTerminalRunLookup interface {
	ListNonTerminalRunIDsByCredential(context.Context, string, int) ([]string, error)
}

type CreateRequest struct {
	CredentialID   string
	LLMGateway     contracts.LLMGatewayConfigRef
	Label          string
	GatewayPolicy  GatewayPolicy
	IdempotencyKey string
	ActorID        string
}

type CreateResult struct {
	Credential Record
	Replayed   bool
}

type DeleteRequest struct {
	CredentialID   string
	IdempotencyKey string
	ActorID        string
}

type DeleteResult struct {
	Replayed bool
}

type ServiceOptions struct {
	Pool     *pgxpool.Pool
	Gateways GatewayLookup
	Managers *ManagerRegistry
	Runs     NonTerminalRunLookup
	Cipher   *TokenCipher
	Now      func() time.Time
	NewID    func(string) (string, error)

	// The two hooks expose only deterministic crash windows to tests. They run
	// after a remote success and before the local atomic commit.
	AfterManagerCreate func() error
	AfterManagerDelete func() error
}

// Service serializes managed credential mutations and is also the Run-create
// guard. The process starts unready and must complete Recover before mutations
// or Run creation are permitted.
type Service struct {
	pool        *pgxpool.Pool
	repository  *Repository
	gateways    GatewayLookup
	managers    *ManagerRegistry
	runs        NonTerminalRunLookup
	cipher      *TokenCipher
	now         func() time.Time
	newID       func(string) (string, error)
	afterCreate func() error
	afterDelete func() error

	lifecycleMu sync.Mutex
	runGuard    sync.RWMutex
	deleteDirty bool
	ready       atomic.Bool
}

func NewService(options ServiceOptions) (*Service, error) {
	if options.Pool == nil || options.Gateways == nil || options.Managers == nil || options.Runs == nil {
		return nil, errors.New("credential lifecycle dependencies are incomplete")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.NewID == nil {
		options.NewID = randomCredentialID
	}
	return &Service{
		pool: options.Pool, repository: NewRepository(options.Pool),
		gateways: options.Gateways, managers: options.Managers, runs: options.Runs,
		cipher: options.Cipher, now: options.Now, newID: options.NewID,
		afterCreate: options.AfterManagerCreate, afterDelete: options.AfterManagerDelete,
	}, nil
}

func (s *Service) ListCredentials(
	ctx context.Context, afterCredentialID string, limit int,
) ([]Record, error) {
	if !s.ready.Load() {
		return nil, ErrRecoveryRequired
	}
	return s.repository.ListCredentials(ctx, afterCredentialID, limit)
}

func (s *Service) GetCredential(ctx context.Context, credentialID string) (Record, error) {
	if !s.ready.Load() {
		return Record{}, ErrRecoveryRequired
	}
	return s.repository.GetCredential(ctx, credentialID)
}

// WithRunCreation holds the shared side of the deletion barrier while a Run
// revalidates credential metadata and commits its immutable snapshot.
func (s *Service) WithRunCreation(ctx context.Context, fn func() error) error {
	if fn == nil {
		return ErrInvalid
	}
	s.runGuard.RLock()
	defer s.runGuard.RUnlock()
	if !s.ready.Load() || s.deleteDirty {
		return ErrRecoveryRequired
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func (s *Service) Recover(ctx context.Context) error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	s.ready.Store(false)
	for {
		operations, err := s.repository.ListPreparedOperations(ctx, maximumCredentialPageSize)
		if err != nil {
			return err
		}
		if len(operations) == 0 {
			s.ready.Store(true)
			return nil
		}
		for _, operation := range operations {
			switch operation.Kind {
			case OperationCreate:
				request, err := decodeCreateOperation(operation)
				if err != nil {
					return err
				}
				if _, err := s.executeCreate(ctx, operation, request, true); err != nil {
					return err
				}
			case OperationDelete:
				request, err := decodeDeleteOperation(operation)
				if err != nil {
					return err
				}
				s.runGuard.Lock()
				s.deleteDirty = true
				err = s.executeDelete(ctx, operation, request)
				if err == nil {
					s.deleteDirty = false
				}
				s.runGuard.Unlock()
				if err != nil {
					return err
				}
			default:
				return errors.New("stored credential operation kind is invalid")
			}
		}
	}
}

func (s *Service) Create(ctx context.Context, request CreateRequest) (CreateResult, error) {
	normalized, requestHash, err := normalizeCreateRequest(request)
	if err != nil {
		return CreateResult{}, err
	}
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	if !s.ready.Load() {
		return CreateResult{}, ErrRecoveryRequired
	}

	prepared, err := s.preparedFor(ctx, OperationCreate, request.IdempotencyKey)
	if err != nil {
		return CreateResult{}, err
	}
	if prepared != nil {
		if prepared.RequestHash != requestHash || prepared.CredentialID != normalized.CredentialID {
			return CreateResult{}, ErrConflict
		}
		storedRequest, err := decodeCreateOperation(*prepared)
		if err != nil || !createOperationRequestsEqual(storedRequest, normalized) {
			return CreateResult{}, ErrConflict
		}
		record, err := s.executeCreate(ctx, *prepared, storedRequest, true)
		return CreateResult{Credential: record, Replayed: err == nil}, err
	}

	existing, err := s.repository.GetOperationByIdempotency(ctx, OperationCreate, request.IdempotencyKey)
	if err == nil {
		if existing.RequestHash != requestHash || existing.CredentialID != normalized.CredentialID {
			return CreateResult{}, ErrConflict
		}
		record, getErr := s.repository.GetCredential(ctx, normalized.CredentialID)
		if getErr != nil {
			return CreateResult{}, ErrConflict
		}
		return CreateResult{Credential: record, Replayed: true}, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return CreateResult{}, err
	}
	if s.cipher == nil {
		return CreateResult{}, ErrKeyUnavailable
	}
	operation, err := s.newOperation(
		OperationCreate, normalized.CredentialID, request.IdempotencyKey, requestHash, normalized,
	)
	if err != nil {
		return CreateResult{}, err
	}
	if err := s.validateCreateManagerRequest(ctx, operation, normalized); err != nil {
		return CreateResult{}, err
	}
	if err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		repository := NewRepository(tx)
		if err := repository.ReserveCredentialID(ctx, normalized.CredentialID, operation.CreatedAt); err != nil {
			return err
		}
		return repository.InsertOperation(ctx, operation)
	}); err != nil {
		return CreateResult{}, err
	}
	record, err := s.executeCreate(ctx, operation, normalized, false)
	return CreateResult{Credential: record}, err
}

func (s *Service) Delete(ctx context.Context, request DeleteRequest) (DeleteResult, error) {
	requestHash, err := validateDeleteRequest(request)
	if err != nil {
		return DeleteResult{}, err
	}
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	if !s.ready.Load() {
		return DeleteResult{}, ErrRecoveryRequired
	}
	s.runGuard.Lock()
	defer s.runGuard.Unlock()

	prepared, err := s.preparedFor(ctx, OperationDelete, request.IdempotencyKey)
	if err != nil {
		return DeleteResult{}, err
	}
	if prepared != nil {
		if prepared.RequestHash != requestHash || prepared.CredentialID != request.CredentialID {
			return DeleteResult{}, ErrConflict
		}
		storedRequest, err := decodeDeleteOperation(*prepared)
		if err != nil || storedRequest.ActorID != request.ActorID {
			return DeleteResult{}, ErrConflict
		}
		s.deleteDirty = true
		err = s.executeDelete(ctx, *prepared, storedRequest)
		if err == nil {
			s.deleteDirty = false
		}
		return DeleteResult{Replayed: err == nil}, err
	}

	existing, err := s.repository.GetOperationByIdempotency(ctx, OperationDelete, request.IdempotencyKey)
	if err == nil {
		if existing.RequestHash != requestHash || existing.CredentialID != request.CredentialID {
			return DeleteResult{}, ErrConflict
		}
		if _, tombstoneErr := s.repository.GetTombstone(ctx, request.CredentialID); tombstoneErr != nil {
			return DeleteResult{}, ErrConflict
		}
		s.deleteDirty = false
		return DeleteResult{Replayed: true}, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return DeleteResult{}, err
	}
	record, err := s.repository.GetCredential(ctx, request.CredentialID)
	if err != nil {
		return DeleteResult{}, err
	}
	runIDs, err := s.runs.ListNonTerminalRunIDsByCredential(
		ctx, request.CredentialID, maximumCredentialRunReferences,
	)
	if err != nil {
		return DeleteResult{}, errors.New("inspect credential Run references")
	}
	if len(runIDs) != 0 {
		return DeleteResult{}, newCredentialInUseError(runIDs)
	}
	if _, _, err := s.resolveManager(record.LLMGateway); err != nil {
		return DeleteResult{}, err
	}
	deletedAt := databaseTime(s.now())
	storedRequest := deleteOperationRequest{
		CredentialID: request.CredentialID, LLMGateway: record.LLMGateway,
		RemoteKeyID: record.RemoteKeyID, ActorID: request.ActorID, DeletedAt: deletedAt,
	}
	operation, err := s.newOperation(
		OperationDelete, request.CredentialID, request.IdempotencyKey, requestHash, storedRequest,
	)
	if err != nil {
		return DeleteResult{}, err
	}
	// From this point until a confirmed completed operation, an ambiguous
	// database or Gateway result must prevent a new Run from pinning this key.
	s.deleteDirty = true
	if err := s.repository.InsertOperation(ctx, operation); err != nil {
		return DeleteResult{}, err
	}
	err = s.executeDelete(ctx, operation, storedRequest)
	if err == nil {
		s.deleteDirty = false
	}
	return DeleteResult{}, err
}

type createOperationRequest struct {
	CredentialID  string                        `json:"credentialId"`
	LLMGateway    contracts.LLMGatewayConfigRef `json:"llmGateway"`
	Label         string                        `json:"label,omitempty"`
	GatewayPolicy GatewayPolicy                 `json:"gatewayPolicy"`
	ActorID       string                        `json:"actorId"`
}

type deleteOperationRequest struct {
	CredentialID string                        `json:"credentialId"`
	LLMGateway   contracts.LLMGatewayConfigRef `json:"llmGateway"`
	RemoteKeyID  string                        `json:"remoteKeyId"`
	ActorID      string                        `json:"actorId"`
	DeletedAt    time.Time                     `json:"deletedAt"`
}

func normalizeCreateRequest(request CreateRequest) (createOperationRequest, string, error) {
	if err := validateCredentialID(request.CredentialID); err != nil || request.LLMGateway.ValidateRef() != nil ||
		!idempotencyKeyPattern.MatchString(request.IdempotencyKey) || !validActorID(request.ActorID) ||
		request.Label != "" && (strings.TrimSpace(request.Label) == "" || !utf8.ValidString(request.Label) ||
			utf8.RuneCountInString(request.Label) > MaximumCredentialLabel) {
		return createOperationRequest{}, "", fmt.Errorf("%w: create credential request is invalid", ErrInvalid)
	}
	if err := request.GatewayPolicy.Validate(); err != nil {
		return createOperationRequest{}, "", err
	}
	policy := request.GatewayPolicy
	policy.ModelPolicies = append([]contracts.ModelPolicyRef(nil), policy.ModelPolicies...)
	policy.MaxBudget = cloneFloat64Pointer(policy.MaxBudget)
	policy.TPMLimit = cloneIntPointer(policy.TPMLimit)
	policy.RPMLimit = cloneIntPointer(policy.RPMLimit)
	policy.MaxParallelRequests = cloneIntPointer(policy.MaxParallelRequests)
	sort.Slice(policy.ModelPolicies, func(left, right int) bool {
		return modelPolicyRefKey(policy.ModelPolicies[left]) < modelPolicyRefKey(policy.ModelPolicies[right])
	})
	normalized := createOperationRequest{
		CredentialID: request.CredentialID, LLMGateway: request.LLMGateway,
		Label: request.Label, GatewayPolicy: policy, ActorID: request.ActorID,
	}
	digest, err := operationRequestDigest(normalized)
	return normalized, digest, err
}

func validateDeleteRequest(request DeleteRequest) (string, error) {
	if err := validateCredentialID(request.CredentialID); err != nil ||
		!idempotencyKeyPattern.MatchString(request.IdempotencyKey) || !validActorID(request.ActorID) {
		return "", fmt.Errorf("%w: delete credential request is invalid", ErrInvalid)
	}
	return operationRequestDigest(struct {
		CredentialID string `json:"credentialId"`
		ActorID      string `json:"actorId"`
	}{CredentialID: request.CredentialID, ActorID: request.ActorID})
}

func (s *Service) executeCreate(
	ctx context.Context,
	operation Operation,
	request createOperationRequest,
	recoverFirst bool,
) (Record, error) {
	gateway, manager, err := s.resolveManager(request.LLMGateway)
	if err != nil {
		return Record{}, err
	}
	if s.cipher == nil {
		return Record{}, ErrKeyUnavailable
	}
	managerRequest := ManagerCreateRequest{
		OperationID: operation.OperationID, CredentialID: request.CredentialID,
		LLMGateway: gateway, Label: request.Label, Policy: request.GatewayPolicy,
	}
	if recoverFirst {
		if err := manager.RecoverCreate(ctx, managerRequest); err != nil {
			return Record{}, ErrGatewayUnavailable
		}
		if err := manager.ValidateCreate(ctx, managerRequest); err != nil {
			return Record{}, safeManagerValidationError(ctx, err)
		}
	}
	generated, err := manager.Create(ctx, managerRequest)
	if err != nil {
		return Record{}, ErrGatewayUnavailable
	}
	if err := generated.Validate(); err != nil {
		return Record{}, ErrGatewayUnavailable
	}
	if s.afterCreate != nil {
		if err := s.afterCreate(); err != nil {
			return Record{}, err
		}
	}
	envelope, err := s.cipher.Seal(request.CredentialID, request.LLMGateway, generated.Token)
	if err != nil {
		return Record{}, err
	}
	effectivePolicy := cloneEffectiveGatewayPolicy(generated.EffectivePolicy)
	record := Record{
		CredentialID: request.CredentialID, LLMGateway: request.LLMGateway,
		RemoteKeyID: generated.RemoteKeyID, Label: request.Label,
		EffectivePolicy: effectivePolicy, Envelope: envelope, CreatedAt: databaseTime(s.now()),
	}
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		repository := NewRepository(tx)
		if err := repository.InsertCredential(ctx, record); err != nil {
			return err
		}
		return repository.CompleteOperation(ctx, operation.OperationID, operationCompletionTime(operation, s.now()))
	})
	if err != nil {
		return Record{}, err
	}
	return record, nil
}

func (s *Service) executeDelete(
	ctx context.Context,
	operation Operation,
	request deleteOperationRequest,
) error {
	if err := validateDeleteOperationRequest(request); err != nil {
		return err
	}
	record, err := s.repository.GetCredential(ctx, request.CredentialID)
	if err != nil || record.LLMGateway != request.LLMGateway || record.RemoteKeyID != request.RemoteKeyID {
		return errors.New("prepared credential deletion no longer matches its active record")
	}
	runIDs, err := s.runs.ListNonTerminalRunIDsByCredential(
		ctx, request.CredentialID, maximumCredentialRunReferences,
	)
	if err != nil {
		return errors.New("inspect credential Run references")
	}
	if len(runIDs) != 0 {
		return newCredentialInUseError(runIDs)
	}
	gateway, manager, err := s.resolveManager(request.LLMGateway)
	if err != nil {
		return err
	}
	if err := manager.Delete(ctx, ManagerDeleteRequest{
		OperationID: operation.OperationID, CredentialID: request.CredentialID,
		LLMGateway: gateway, RemoteKeyID: request.RemoteKeyID,
	}); err != nil {
		return ErrGatewayUnavailable
	}
	if s.afterDelete != nil {
		if err := s.afterDelete(); err != nil {
			return err
		}
	}
	return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		repository := NewRepository(tx)
		if err := repository.DeleteCredential(ctx, request.CredentialID); err != nil {
			return err
		}
		if err := repository.InsertTombstone(ctx, Tombstone{
			CredentialID: request.CredentialID, ActorID: request.ActorID, DeletedAt: request.DeletedAt,
		}); err != nil {
			return err
		}
		return repository.CompleteOperation(ctx, operation.OperationID, operationCompletionTime(operation, s.now()))
	})
}

func (s *Service) resolveManager(
	ref contracts.LLMGatewayConfigRef,
) (contracts.ResolvedLLMGatewayConfig, GatewayCredentialManager, error) {
	gateway, err := s.gateways.LLMGateway(ref.GatewayID + "@" + ref.Version)
	if err != nil || gateway.Ref != ref {
		return contracts.ResolvedLLMGatewayConfig{}, nil, fmt.Errorf("%w: exact LLM Gateway is unavailable", ErrInvalid)
	}
	manager, err := s.managers.ForGateway(gateway)
	if err != nil {
		return contracts.ResolvedLLMGatewayConfig{}, nil, ErrManagerUnavailable
	}
	return gateway, manager, nil
}

func (s *Service) validateCreateManagerRequest(
	ctx context.Context,
	operation Operation,
	request createOperationRequest,
) error {
	gateway, manager, err := s.resolveManager(request.LLMGateway)
	if err != nil {
		return err
	}
	err = manager.ValidateCreate(ctx, ManagerCreateRequest{
		OperationID: operation.OperationID, CredentialID: request.CredentialID,
		LLMGateway: gateway, Label: request.Label, Policy: request.GatewayPolicy,
	})
	if err != nil {
		return safeManagerValidationError(ctx, err)
	}
	return nil
}

func safeManagerValidationError(ctx context.Context, err error) error {
	if contextError := ctx.Err(); contextError != nil {
		return contextError
	}
	switch {
	case errors.Is(err, ErrInvalid):
		return ErrInvalid
	case errors.Is(err, ErrManagerUnavailable):
		return ErrManagerUnavailable
	case errors.Is(err, ErrGatewayUnavailable):
		return ErrGatewayUnavailable
	default:
		return errors.New("validate Gateway credential request")
	}
}

func (s *Service) preparedFor(
	ctx context.Context, kind OperationKind, idempotencyKey string,
) (*Operation, error) {
	operations, err := s.repository.ListPreparedOperations(ctx, 2)
	if err != nil {
		return nil, err
	}
	if len(operations) == 0 {
		return nil, nil
	}
	if len(operations) != 1 || operations[0].Kind != kind || operations[0].IdempotencyKey != idempotencyKey {
		return nil, ErrRecoveryRequired
	}
	return &operations[0], nil
}

func (s *Service) newOperation(
	kind OperationKind,
	credentialID, idempotencyKey, requestHash string,
	request any,
) (Operation, error) {
	operationID, err := s.newID("credop_")
	if err != nil {
		return Operation{}, errors.New("generate credential operation ID")
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return Operation{}, fmt.Errorf("%w: encode credential operation", ErrInvalid)
	}
	now := databaseTime(s.now())
	operation := Operation{
		OperationID: operationID, IdempotencyKey: idempotencyKey, RequestHash: requestHash,
		CredentialID: credentialID, Kind: kind, Phase: OperationPrepared,
		Request: encoded, CreatedAt: now, UpdatedAt: now,
	}
	if err := validateOperation(operation); err != nil {
		return Operation{}, err
	}
	return operation, nil
}

func decodeCreateOperation(operation Operation) (createOperationRequest, error) {
	var request createOperationRequest
	if operation.Kind != OperationCreate || decodeStrictJSON(operation.Request, &request) != nil {
		return createOperationRequest{}, errors.New("stored create-credential operation is invalid")
	}
	if err := validateCredentialID(request.CredentialID); err != nil || request.CredentialID != operation.CredentialID ||
		request.LLMGateway.ValidateRef() != nil || !validActorID(request.ActorID) ||
		request.Label != "" && (strings.TrimSpace(request.Label) == "" || !utf8.ValidString(request.Label) ||
			utf8.RuneCountInString(request.Label) > MaximumCredentialLabel) || request.GatewayPolicy.Validate() != nil {
		return createOperationRequest{}, errors.New("stored create-credential operation is invalid")
	}
	for index := 1; index < len(request.GatewayPolicy.ModelPolicies); index++ {
		if modelPolicyRefKey(request.GatewayPolicy.ModelPolicies[index-1]) >=
			modelPolicyRefKey(request.GatewayPolicy.ModelPolicies[index]) {
			return createOperationRequest{}, errors.New("stored create-credential operation is not canonical")
		}
	}
	digest, err := operationRequestDigest(request)
	if err != nil || digest != operation.RequestHash {
		return createOperationRequest{}, errors.New("stored create-credential operation integrity check failed")
	}
	return request, nil
}

func decodeDeleteOperation(operation Operation) (deleteOperationRequest, error) {
	var request deleteOperationRequest
	if operation.Kind != OperationDelete || decodeStrictJSON(operation.Request, &request) != nil ||
		request.CredentialID != operation.CredentialID || validateDeleteOperationRequest(request) != nil {
		return deleteOperationRequest{}, errors.New("stored delete-credential operation is invalid")
	}
	digest, err := operationRequestDigest(struct {
		CredentialID string `json:"credentialId"`
		ActorID      string `json:"actorId"`
	}{CredentialID: request.CredentialID, ActorID: request.ActorID})
	if err != nil || digest != operation.RequestHash {
		return deleteOperationRequest{}, errors.New("stored delete-credential operation integrity check failed")
	}
	return request, nil
}

func validateDeleteOperationRequest(request deleteOperationRequest) error {
	if validateCredentialID(request.CredentialID) != nil || request.LLMGateway.ValidateRef() != nil ||
		strings.TrimSpace(request.RemoteKeyID) == "" || len(request.RemoteKeyID) > MaximumRemoteKeyIDBytes ||
		!utf8.ValidString(request.RemoteKeyID) || !validActorID(request.ActorID) || request.DeletedAt.IsZero() {
		return errors.New("stored delete-credential operation is invalid")
	}
	return nil
}

func operationRequestDigest(request any) (string, error) {
	encoded, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("%w: credential request cannot be encoded", ErrInvalid)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func createOperationRequestsEqual(left, right createOperationRequest) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func newCredentialInUseError(runIDs []string) *CredentialInUseError {
	result := append([]string(nil), runIDs...)
	sort.Strings(result)
	if len(result) > maximumCredentialRunReferences {
		result = result[:maximumCredentialRunReferences]
	}
	return &CredentialInUseError{RunIDs: result}
}

func modelPolicyRefKey(ref contracts.ModelPolicyRef) string {
	return ref.PolicyID + "\x00" + ref.Version + "\x00" + ref.Digest
}

func cloneEffectiveGatewayPolicy(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
	value.ModelPolicies = append([]contracts.ModelPolicyRef(nil), value.ModelPolicies...)
	value.Models = append([]string(nil), value.Models...)
	value.MaxBudget = cloneFloat64Pointer(value.MaxBudget)
	value.TPMLimit = cloneIntPointer(value.TPMLimit)
	value.RPMLimit = cloneIntPointer(value.RPMLimit)
	value.MaxParallelRequests = cloneIntPointer(value.MaxParallelRequests)
	return value
}

func cloneFloat64Pointer(value *float64) *float64 {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func cloneIntPointer(value *int) *int {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func operationCompletionTime(operation Operation, candidate time.Time) time.Time {
	result := databaseTime(candidate)
	floor := databaseTime(operation.UpdatedAt)
	if result.Before(floor) {
		return floor
	}
	return result
}

func validActorID(value string) bool {
	return strings.TrimSpace(value) != "" && len(value) <= 256 && utf8.ValidString(value)
}

func randomCredentialID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := cryptorand.Read(buffer); err != nil {
		return "", errors.New("generate random credential operation ID")
	}
	return prefix + hex.EncodeToString(buffer), nil
}
