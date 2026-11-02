package credentials

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const maximumCredentialRunReferences = 128

var (
	ErrGatewayUnavailable = apierror.New(apierror.GatewayUnavailable, "Gateway credential operation is unavailable")
	ErrRecoveryRequired   = apierror.New(apierror.CredentialRecoveryRequired, "credential operation recovery is required")
	// ErrCreateAbandoned reports a create whose idempotency key belongs to an
	// operation that recovery abandoned: its remote alias was confirmed absent
	// and its stored request no longer validates. The credential ID stays
	// reserved; a new request needs a new credential ID and idempotency key.
	ErrCreateAbandoned = fmt.Errorf(
		"%w: credential create operation was abandoned because its request is no longer valid", ErrConflict,
	)
	errCreateKeyReused = fmt.Errorf("%w: LLM credential create key", contracts.ErrIdempotencyConflict)
	errDeleteKeyReused = fmt.Errorf("%w: LLM credential delete key", contracts.ErrIdempotencyConflict)
	// errLeaseLost refuses a managed credential write of a Server that no
	// longer holds the advisory lease (PER-R34). Nothing of the write
	// commits; a prepared operation it leaves belongs to the next active
	// Server's startup recovery, so the answer is the retryable
	// credential_recovery_required.
	errLeaseLost = fmt.Errorf("%w: %w", ErrRecoveryRequired, persistencepostgres.ErrLeaseLost)
)

type CredentialInUseError struct {
	RunIDs        []string
	AuditIDs      []string
	BindingLabels []string
}

func (e *CredentialInUseError) Error() string {
	return "LLM credential is referenced by a non-terminal WorkflowRun, Audit dispatch hold or active RuntimeConfig label"
}

// ErrorKind answers the error with credential_in_use; its references are the
// details.
func (e *CredentialInUseError) ErrorKind() apierror.Kind { return apierror.CredentialInUse }

type GatewayLookup interface {
	LLMGateway(string) (llmgateway.ResolvedLLMGatewayConfig, error)
}

type NonTerminalRunLookup interface {
	ListNonTerminalRunIDsByCredential(context.Context, string, int) ([]string, error)
}

type AuditDispatchHoldLookup interface {
	ListHeldAuditIDsByLLMCredential(context.Context, string, int) ([]string, error)
}

type CreateRequest struct {
	CredentialID   string
	LLMGateway     llmgateway.LLMGatewayConfigRef
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
	Pool *pgxpool.Pool
	// Lease is the active Server's advisory lease. Every write of the
	// operation journal and of managed credential records proves it in its
	// own transaction (PER-R34).
	Lease    persistencepostgres.ServerLease
	Gateways GatewayLookup
	Managers *ManagerRegistry
	Runs     NonTerminalRunLookup
	Audits   AuditDispatchHoldLookup
	Cipher   *TokenCipher
	Barrier  *LifecycleBarrier
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
	lease       persistencepostgres.ServerLease
	repository  *Repository
	gateways    GatewayLookup
	managers    *ManagerRegistry
	runs        NonTerminalRunLookup
	audits      AuditDispatchHoldLookup
	cipher      *TokenCipher
	now         func() time.Time
	newID       func(string) (string, error)
	afterCreate func() error
	afterDelete func() error

	lifecycleMu sync.Mutex
	barrier     *LifecycleBarrier
	// uncertainDelete names the credential of a delete whose prepared intent
	// may or may not be durable: the transaction that inserted it failed
	// without a definite outcome. While it is set, both reference guards
	// refuse every new reference, because the per-credential prepared-delete
	// lookup cannot see an intent that is still committing. A durable
	// prepared delete otherwise fences only its own credential ID, through
	// EncryptedProvider lookups, instead of every Run creation.
	//
	// Only Delete records it, under lifecycleMu and the barrier's write side.
	// settleUncertainDeleteLocked clears it once durable state proves that no
	// prepared deletion of the credential remains, and a completed delete of
	// that credential clears it. fenceMu serializes reconciliation with every
	// write; a guard reads the pointer without a lock, so the clean guard does
	// no extra work. Lock order: lifecycleMu, then the barrier, then fenceMu.
	fenceMu         sync.Mutex
	uncertainDelete atomic.Pointer[uncertainDelete]
	ready           atomic.Bool
}

// Format redacts the service as a whole for every verb (SEC-R88): through its
// unexported cipher field fmt would print the cipher's key bytes.
func (s *Service) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "credentials.Service([REDACTED])")
}

// uncertainDelete identifies a delete intent whose durable outcome is unknown.
type uncertainDelete struct {
	credentialID string
}

// deleteIntentState is what durable state says about an uncertain delete.
type deleteIntentState int

const (
	// deleteIntentResolved: no prepared deletion of the credential remains,
	// so the intent never became durable or its deletion has completed.
	deleteIntentResolved deleteIntentState = iota
	// deleteIntentPrepared: a prepared deletion of the credential is durable;
	// the same request or recovery must complete it.
	deleteIntentPrepared
	// deleteIntentUnsettled: the transaction that inserted the intent, or a
	// reference to the credential, may still hold its deletion lock, so a
	// negative read would prove nothing.
	deleteIntentUnsettled
)

func NewService(options ServiceOptions) (*Service, error) {
	if options.Pool == nil || !options.Lease.Valid() || options.Gateways == nil || options.Managers == nil ||
		options.Runs == nil {
		return nil, errors.New("credential lifecycle dependencies are incomplete")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.NewID == nil {
		options.NewID = randomid.New
	}
	if options.Barrier == nil {
		options.Barrier = NewLifecycleBarrier()
	}
	return &Service{
		pool: options.Pool, lease: options.Lease, repository: NewRepository(options.Pool),
		gateways: options.Gateways, managers: options.Managers, runs: options.Runs, audits: options.Audits,
		cipher: options.Cipher, barrier: options.Barrier, now: options.Now, newID: options.NewID,
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
// revalidates credential metadata and commits its immutable snapshot. A
// credential with a prepared delete is rejected by its metadata lookup, so
// such a delete does not block Runs that pin other credentials. While a
// delete's outcome is uncertain, every reference waits for its durable
// reconciliation.
func (s *Service) WithRunCreation(ctx context.Context, fn func() error) error {
	if fn == nil {
		return ErrInvalid
	}
	return s.barrier.WithCredentialReferences(ctx, func() error {
		if !s.ready.Load() {
			return ErrRecoveryRequired
		}
		if s.uncertainDelete.Load() != nil {
			if err := s.reconcileForReferences(ctx); err != nil {
				return err
			}
		}
		return fn()
	})
}

// reconcileForReferences lets a reference guard clear an uncertain delete
// that durable state has resolved. The caller holds the barrier's shared
// side, so no Delete can record another intent meanwhile; a concurrent guard
// waits for the reconciliation in progress and then finds the fence cleared
// or reconciles again. The work is one short transaction that never waits for
// a database lock, whatever the size of the operation journal.
func (s *Service) reconcileForReferences(ctx context.Context) error {
	s.fenceMu.Lock()
	defer s.fenceMu.Unlock()
	state, err := s.settleUncertainDeleteLocked(ctx)
	if err != nil {
		return reconciliationFailure(ctx, err)
	}
	if state != deleteIntentResolved {
		return ErrRecoveryRequired
	}
	return nil
}

// errReconciliationUnreadable reports a reconciliation that could not read
// durable state. Its public outcome is the retryable credential_recovery_required.
var errReconciliationUnreadable = fmt.Errorf(
	"%w: the uncertain credential deletion could not be reconciled with durable state", ErrRecoveryRequired,
)

// reconciliationFailure answers a reconciliation that failed for a guard or a
// delete. The fence stays closed either way. A live caller gets the retryable
// credential_recovery_required of RCF-R53 rather than a storage failure; the
// storage cause is not kept, so a timeout inside the database read cannot pass
// for the caller's own cancellation. A cancelled or expired caller keeps its
// context error.
func reconciliationFailure(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return persistencepostgres.WrapError("reconcile uncertain LLM credential deletion", err)
	}
	return errReconciliationUnreadable
}

// settleUncertainDeleteLocked reconciles the uncertain delete, if any, with
// durable state and clears it when no prepared deletion of its credential
// remains. The caller holds fenceMu.
//
// The transaction that inserted the intent held the credential's deletion
// lock from before its INSERT until it ended, so only after this transaction
// takes that lock is a negative read proof that the intent did not commit; a
// bare negative read could race a COMMIT still in flight. The lock is tried,
// never awaited, so a caller does not queue behind an open transaction. The
// read is a later statement of a READ COMMITTED transaction, so its snapshot
// is taken after the lock and sees whatever that transaction committed.
func (s *Service) settleUncertainDeleteLocked(ctx context.Context) (deleteIntentState, error) {
	intent := s.uncertainDelete.Load()
	if intent == nil {
		return deleteIntentResolved, nil
	}
	state := deleteIntentUnsettled
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.ReadCommitted}, func(tx pgx.Tx) error {
		locked, err := runtimeconfig.TryLockLLMCredentialDeletion(ctx, tx, intent.credentialID)
		if err != nil || !locked {
			return err
		}
		prepared, err := NewRepository(tx).HasPreparedDelete(ctx, intent.credentialID)
		if err != nil {
			return err
		}
		state = deleteIntentResolved
		if prepared {
			state = deleteIntentPrepared
		}
		return nil
	})
	if err != nil {
		return deleteIntentUnsettled, err
	}
	if state == deleteIntentResolved {
		s.uncertainDelete.Store(nil)
	}
	return state, nil
}

// recordUncertainDelete fences every new reference until reconciliation
// settles a delete intent whose insert or commit outcome is unknown. Delete
// settles any earlier intent before it prepares another, so at most one is
// outstanding.
func (s *Service) recordUncertainDelete(credentialID string) {
	s.fenceMu.Lock()
	defer s.fenceMu.Unlock()
	s.uncertainDelete.Store(&uncertainDelete{credentialID: credentialID})
}

// resolveUncertainDelete clears the uncertain delete of credentialID after a
// delete of that credential completed: at most one deletion of a credential
// is prepared at a time, so none of it remains.
func (s *Service) resolveUncertainDelete(credentialID string) {
	s.fenceMu.Lock()
	defer s.fenceMu.Unlock()
	if intent := s.uncertainDelete.Load(); intent != nil && intent.credentialID == credentialID {
		s.uncertainDelete.Store(nil)
	}
}

// WithAllocationReferences uses the same recovery-aware shared fence as Run
// creation. Placement invokes it around the transaction that introduces live
// allocation references, so a managed key deletion cannot pass between
// authorization revalidation and commit.
func (s *Service) WithAllocationReferences(ctx context.Context, fn func() error) error {
	return s.WithRunCreation(ctx, fn)
}

// Recover completes every prepared operation of the journal before it marks
// the Service ready. The Server runs it only after the lease handover, so no
// journal write of a previous lease owner can still commit behind its reads
// (PER-R34, RCF-R72). An empty journal does not prove that an uncertain delete
// of this process left no intent, because its transaction may still commit;
// Recover settles that delete through durable reconciliation, completes an
// intent that turns out durable, and keeps the fence while the outcome is
// still unsettled.
func (s *Service) Recover(ctx context.Context) error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	s.ready.Store(false)
	settledPrepared := false
	for {
		operations, err := s.repository.ListPreparedOperations(ctx, maximumPreparedOperationBatch)
		if err != nil {
			return err
		}
		if len(operations) == 0 {
			s.fenceMu.Lock()
			state, err := s.settleUncertainDeleteLocked(ctx)
			s.fenceMu.Unlock()
			if err != nil {
				return persistencepostgres.WrapError("reconcile uncertain LLM credential deletion", err)
			}
			if state == deleteIntentPrepared {
				// The intent committed after the journal was read; the next
				// pass completes it. A second such answer after an empty read
				// means the journal reads disagree, which recovery must not
				// paper over.
				if settledPrepared {
					return errors.New("prepared credential deletion is missing from the operation journal")
				}
				settledPrepared = true
				continue
			}
			s.ready.Store(true)
			return nil
		}
		settledPrepared = false
		for _, operation := range operations {
			switch operation.Kind {
			case OperationCreate:
				request, err := decodeCreateOperation(operation)
				if err != nil {
					return err
				}
				if _, err := s.executeCreate(ctx, operation, request, true); err != nil &&
					!errors.Is(err, ErrCreateAbandoned) {
					return err
				}
			case OperationDelete:
				request, err := decodeDeleteOperation(operation)
				if err != nil {
					return err
				}
				if err := s.barrier.lockMutation(ctx); err != nil {
					return err
				}
				err = s.executeDelete(ctx, operation, request)
				s.barrier.unlockMutation()
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
	// Without the master key no credential row can exist (RCF-R44), so the
	// request stops before any read, write or Gateway call.
	if s.cipher == nil {
		return CreateResult{}, ErrCredentialKeyUnavailable
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
			return CreateResult{}, errCreateKeyReused
		}
		storedRequest, err := decodeCreateOperation(*prepared)
		if err != nil {
			return CreateResult{}, ErrConflict
		}
		if !createOperationRequestsEqual(storedRequest, normalized) {
			return CreateResult{}, errCreateKeyReused
		}
		record, err := s.executeCreate(ctx, *prepared, storedRequest, true)
		return CreateResult{Credential: record}, err
	}

	existing, err := s.repository.GetOperationByIdempotency(ctx, OperationCreate, request.IdempotencyKey)
	if err == nil {
		if existing.RequestHash != requestHash || existing.CredentialID != normalized.CredentialID {
			return CreateResult{}, errCreateKeyReused
		}
		if existing.Phase == OperationAbandoned {
			return CreateResult{}, ErrCreateAbandoned
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
		if err := s.proveLease(ctx, tx); err != nil {
			return err
		}
		// The create operation reserves the ID forever: a unique index admits one
		// create per credential ID, and operations are never deleted.
		return NewRepository(tx).InsertOperation(ctx, operation)
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
	if s.cipher == nil {
		return DeleteResult{}, ErrCredentialKeyUnavailable
	}
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	if !s.ready.Load() {
		return DeleteResult{}, ErrRecoveryRequired
	}
	if err := s.barrier.lockMutation(ctx); err != nil {
		return DeleteResult{}, err
	}
	defer s.barrier.unlockMutation()

	prepared, err := s.preparedFor(ctx, OperationDelete, request.IdempotencyKey)
	if err != nil {
		return DeleteResult{}, err
	}
	if prepared != nil {
		if prepared.RequestHash != requestHash || prepared.CredentialID != request.CredentialID {
			return DeleteResult{}, errDeleteKeyReused
		}
		storedRequest, err := decodeDeleteOperation(*prepared)
		if err != nil {
			return DeleteResult{}, ErrConflict
		}
		if storedRequest.ActorID != request.ActorID {
			return DeleteResult{}, errDeleteKeyReused
		}
		err = s.executeDelete(ctx, *prepared, storedRequest)
		if err == nil {
			s.resolveUncertainDelete(prepared.CredentialID)
		}
		return DeleteResult{}, err
	}

	existing, err := s.repository.GetOperationByIdempotency(ctx, OperationDelete, request.IdempotencyKey)
	if err == nil {
		if existing.RequestHash != requestHash || existing.CredentialID != request.CredentialID {
			return DeleteResult{}, errDeleteKeyReused
		}
		// A completed delete operation is the credential's tombstone. Its
		// replay proves nothing about another delete's uncertain intent.
		if existing.Phase != OperationCompleted {
			return DeleteResult{}, ErrConflict
		}
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
		return DeleteResult{}, persistencepostgres.WrapError("inspect credential Run references", err)
	}
	auditIDs, err := s.listAuditHolds(ctx, request.CredentialID)
	if err != nil {
		return DeleteResult{}, err
	}
	// An earlier delete whose intent may still commit is settled before
	// another intent is prepared: the journal read above cannot see an
	// uncommitted intent, and at most one uncertain delete is tracked.
	s.fenceMu.Lock()
	earlier, err := s.settleUncertainDeleteLocked(ctx)
	s.fenceMu.Unlock()
	if err != nil {
		return DeleteResult{}, reconciliationFailure(ctx, err)
	}
	if earlier != deleteIntentResolved {
		return DeleteResult{}, ErrRecoveryRequired
	}
	// Once the prepared intent is durable, lookups of this credential fail
	// until the operation completes. If the insert or commit outcome is
	// unknown, fence every new reference until reconciliation proves that no
	// prepared deletion remains or a delete of the credential completes.
	var operation Operation
	var storedRequest deleteOperationRequest
	attemptedInsert := false
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := runtimeconfig.LockLLMCredentialDeletion(ctx, tx, request.CredentialID); err != nil {
			return err
		}
		repository := NewRepository(tx)
		current, err := repository.GetCredential(ctx, request.CredentialID)
		if err != nil {
			return err
		}
		if current.LLMGateway != record.LLMGateway || current.RemoteKeyID != record.RemoteKeyID {
			return ErrConflict
		}
		prepared, err := repository.HasPreparedDelete(ctx, request.CredentialID)
		if err != nil {
			return err
		}
		if prepared {
			return ErrRecoveryRequired
		}
		labels, err := repository.ListActiveRuntimeBindingLabelsByCredential(
			ctx, request.CredentialID, maximumCredentialRunReferences,
		)
		if err != nil {
			return err
		}
		if len(runIDs) != 0 || len(auditIDs) != 0 || len(labels) != 0 {
			return newCredentialInUseError(runIDs, auditIDs, labels)
		}
		if _, _, err := s.resolveManager(record.LLMGateway); err != nil {
			return err
		}
		storedRequest = deleteOperationRequest{
			CredentialID: request.CredentialID, LLMGateway: record.LLMGateway,
			RemoteKeyID: record.RemoteKeyID, ActorID: request.ActorID, DeletedAt: persistencepostgres.Timestamp(s.now()),
		}
		operation, err = s.newOperation(
			OperationDelete, request.CredentialID, request.IdempotencyKey, requestHash, storedRequest,
		)
		if err != nil {
			return err
		}
		// A Server that lost the lease records no intent: the next active
		// Server read the journal once its handover ended (PER-R34).
		if err := s.proveLease(ctx, tx); err != nil {
			return err
		}
		attemptedInsert = true
		return repository.InsertOperation(ctx, operation)
	})
	if err != nil {
		var inUse *CredentialInUseError
		if attemptedInsert && !errors.Is(err, ErrConflict) && !errors.Is(err, ErrInvalid) && !errors.As(err, &inUse) {
			s.recordUncertainDelete(request.CredentialID)
		}
		return DeleteResult{}, err
	}
	return DeleteResult{}, s.executeDelete(ctx, operation, storedRequest)
}

type createOperationRequest struct {
	CredentialID  string                         `json:"credentialId"`
	LLMGateway    llmgateway.LLMGatewayConfigRef `json:"llmGateway"`
	Label         string                         `json:"label,omitempty"`
	GatewayPolicy GatewayPolicy                  `json:"gatewayPolicy"`
	ActorID       string                         `json:"actorId"`
}

type deleteOperationRequest struct {
	CredentialID string                         `json:"credentialId"`
	LLMGateway   llmgateway.LLMGatewayConfigRef `json:"llmGateway"`
	RemoteKeyID  string                         `json:"remoteKeyId"`
	ActorID      string                         `json:"actorId"`
	DeletedAt    time.Time                      `json:"deletedAt"`
}

func normalizeCreateRequest(request CreateRequest) (createOperationRequest, string, error) {
	if err := validateCredentialID(request.CredentialID); err != nil || request.LLMGateway.ValidateRef() != nil ||
		!contracts.ValidIdempotencyKey(request.IdempotencyKey) || !validActorID(request.ActorID) ||
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
	policy.TPMLimit = clone.Pointer(policy.TPMLimit)
	policy.RPMLimit = clone.Pointer(policy.RPMLimit)
	policy.MaxParallelRequests = clone.Pointer(policy.MaxParallelRequests)
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
		!contracts.ValidIdempotencyKey(request.IdempotencyKey) || !validActorID(request.ActorID) {
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
		return Record{}, ErrCredentialKeyUnavailable
	}
	managerRequest := ManagerCreateRequest{
		OperationID: operation.OperationID, CredentialID: request.CredentialID,
		LLMGateway: gateway, Label: request.Label, Policy: request.GatewayPolicy,
	}
	if recoverFirst {
		// RecoverCreate removes the deterministic remote alias, which may by
		// now name the key that the next active Server created when it
		// recovered this operation. A Server that lost the lease therefore
		// makes no such call; only a pause between this proof and the call
		// remains (PER-R36).
		if err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return s.proveLease(ctx, tx)
		}); err != nil {
			return Record{}, err
		}
		if err := manager.RecoverCreate(ctx, managerRequest); err != nil {
			return Record{}, ErrGatewayUnavailable
		}
		if err := manager.ValidateCreate(ctx, managerRequest); err != nil {
			validationErr := safeManagerValidationError(ctx, err)
			if !errors.Is(validationErr, ErrInvalid) {
				return Record{}, validationErr
			}
			// RecoverCreate confirmed that the deterministic remote alias is
			// absent, and validation failed deterministically, so no replay can
			// ever succeed. Record a terminal failure instead of blocking
			// recovery forever. Ambiguous failures above stay prepared.
			if err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				if err := s.proveLease(ctx, tx); err != nil {
					return err
				}
				return NewRepository(tx).AbandonOperation(
					ctx, operation.OperationID, operationCompletionTime(operation, s.now()),
				)
			}); err != nil {
				return Record{}, err
			}
			return Record{}, ErrCreateAbandoned
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
		EffectivePolicy: effectivePolicy, Envelope: envelope, CreatedAt: persistencepostgres.Timestamp(s.now()),
	}
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := s.proveLease(ctx, tx); err != nil {
			return err
		}
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
	if err != nil {
		return persistencepostgres.WrapError("read prepared credential deletion record", err)
	}
	if record.LLMGateway != request.LLMGateway || record.RemoteKeyID != request.RemoteKeyID {
		return errors.New("prepared credential deletion no longer matches its active record")
	}
	runIDs, err := s.runs.ListNonTerminalRunIDsByCredential(
		ctx, request.CredentialID, maximumCredentialRunReferences,
	)
	if err != nil {
		return persistencepostgres.WrapError("inspect credential Run references", err)
	}
	auditIDs, err := s.listAuditHolds(ctx, request.CredentialID)
	if err != nil {
		return err
	}
	labels, err := s.repository.ListActiveRuntimeBindingLabelsByCredential(
		ctx, request.CredentialID, maximumCredentialRunReferences,
	)
	if err != nil {
		return err
	}
	if len(runIDs) != 0 || len(auditIDs) != 0 || len(labels) != 0 {
		return newCredentialInUseError(runIDs, auditIDs, labels)
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
		if err := s.proveLease(ctx, tx); err != nil {
			return err
		}
		repository := NewRepository(tx)
		if err := repository.DeleteCredential(ctx, request.CredentialID); err != nil {
			return err
		}
		// Completing the operation records the tombstone: its request keeps the
		// actor and deletion time.
		return repository.CompleteOperation(ctx, operation.OperationID, operationCompletionTime(operation, s.now()))
	})
}

// proveLease fences one write of the operation journal or of a managed
// credential record by the advisory lease, in the write's own transaction
// (PER-R34). The next active Server reads the journal only once every such
// transaction of its predecessor has ended, and a write that starts after
// that cannot prove the lease and commits nothing.
func (s *Service) proveLease(ctx context.Context, tx pgx.Tx) error {
	err := s.lease.Prove(ctx, tx)
	if errors.Is(err, persistencepostgres.ErrLeaseLost) {
		return errLeaseLost
	}
	return err
}

func (s *Service) listAuditHolds(ctx context.Context, credentialID string) ([]string, error) {
	if s.audits == nil {
		return nil, nil
	}
	auditIDs, err := s.audits.ListHeldAuditIDsByLLMCredential(
		ctx, credentialID, maximumCredentialRunReferences,
	)
	if err != nil {
		return nil, persistencepostgres.WrapError("inspect credential Audit references", err)
	}
	return auditIDs, nil
}

func (s *Service) resolveManager(
	ref llmgateway.LLMGatewayConfigRef,
) (llmgateway.ResolvedLLMGatewayConfig, GatewayCredentialManager, error) {
	gateway, err := s.gateways.LLMGateway(ref.GatewayID + "@" + ref.Version)
	if err != nil || gateway.Ref != ref {
		return llmgateway.ResolvedLLMGatewayConfig{}, nil, fmt.Errorf("%w: exact LLM Gateway is unavailable", ErrInvalid)
	}
	manager, err := s.managers.ForGateway(gateway)
	if err != nil {
		return llmgateway.ResolvedLLMGatewayConfig{}, nil, ErrManagerUnavailable
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
	now := persistencepostgres.Timestamp(s.now())
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
	if operation.Kind != OperationCreate || strictjson.Decode(operation.Request, &request) != nil {
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
	if operation.Kind != OperationDelete || strictjson.Decode(operation.Request, &request) != nil ||
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
	digest, err := contentdigest.JSON(request)
	if err != nil {
		return "", fmt.Errorf("%w: credential request cannot be encoded", ErrInvalid)
	}
	return digest, nil
}

func createOperationRequestsEqual(left, right createOperationRequest) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func newCredentialInUseError(runIDs, auditIDs, bindingLabels []string) *CredentialInUseError {
	runs := append([]string{}, runIDs...)
	sort.Strings(runs)
	if len(runs) > maximumCredentialRunReferences {
		runs = runs[:maximumCredentialRunReferences]
	}
	audits := append([]string(nil), auditIDs...)
	sort.Strings(audits)
	if len(audits) > maximumCredentialRunReferences {
		audits = audits[:maximumCredentialRunReferences]
	}
	labels := append([]string(nil), bindingLabels...)
	sort.Strings(labels)
	if len(labels) > maximumCredentialRunReferences {
		labels = labels[:maximumCredentialRunReferences]
	}
	return &CredentialInUseError{RunIDs: runs, AuditIDs: audits, BindingLabels: labels}
}

func modelPolicyRefKey(ref contracts.ModelPolicyRef) string {
	return ref.PolicyID + "\x00" + ref.Version + "\x00" + ref.Digest
}

func cloneEffectiveGatewayPolicy(value EffectiveGatewayPolicy) EffectiveGatewayPolicy {
	value.ModelPolicies = append([]contracts.ModelPolicyRef(nil), value.ModelPolicies...)
	value.Models = append([]string(nil), value.Models...)
	value.MaxBudget = cloneFloat64Pointer(value.MaxBudget)
	value.TPMLimit = clone.Pointer(value.TPMLimit)
	value.RPMLimit = clone.Pointer(value.RPMLimit)
	value.MaxParallelRequests = clone.Pointer(value.MaxParallelRequests)
	return value
}

func cloneFloat64Pointer(value *float64) *float64 {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func operationCompletionTime(operation Operation, candidate time.Time) time.Time {
	result := persistencepostgres.Timestamp(candidate)
	floor := persistencepostgres.Timestamp(operation.UpdatedAt)
	if result.Before(floor) {
		return floor
	}
	return result
}

func validActorID(value string) bool {
	return strings.TrimSpace(value) != "" && len(value) <= 256 && utf8.ValidString(value)
}
