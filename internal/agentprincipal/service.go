package agentprincipal

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"slices"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	labelsReplaceOperation = "runtime-agent.labels-replace"
	deleteOperation        = "runtime-agent.delete"
)

// AdapterRow is one listed principal with the Runtime adapters its labels
// require.
type AdapterRow struct {
	Principal               Principal
	RequiredRuntimeAdapters []string
}

type MutationResult struct {
	Principal *Principal
	Deleted   bool
	Replayed  bool
}

type mutationResultWire struct {
	Principal *Principal `json:"principal,omitempty"`
	Deleted   bool       `json:"deleted,omitempty"`
}

// DeletionGuard keeps a principal from registering again while its delete
// runs; the Control Plane registry implements it.
type DeletionGuard interface {
	BeginPrincipalDeletion(string) (func(), error)
}

type ServiceOptions struct {
	Pool          *pgxpool.Pool
	DeletionGuard DeletionGuard
	Now           func() time.Time
}

type Service struct {
	pool          *pgxpool.Pool
	repository    *Repository
	deletionGuard DeletionGuard
	now           func() time.Time
}

func NewService(options ServiceOptions) (*Service, error) {
	if options.Pool == nil {
		return nil, errors.New("Runtime Agent principal service requires PostgreSQL")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &Service{
		pool: options.Pool, repository: NewRepository(options.Pool),
		deletionGuard: options.DeletionGuard, now: options.Now,
	}, nil
}

// Register seeds labels only for a previously unseen certificate-key
// principal. Existing principals are returned before mutable binding lookup.
func (s *Service) Register(ctx context.Context, runtimeAgentID string, initialLabels []string) (Principal, error) {
	if err := validateRuntimeAgentID(runtimeAgentID); err != nil || runtimeconfig.ValidateAgentLabels(initialLabels) != nil {
		return Principal{}, invalid("Runtime Agent principal registration is invalid")
	}
	existing, err := s.repository.Get(ctx, runtimeAgentID)
	if err == nil {
		return existing, nil
	}
	if !errors.Is(err, runtimeconfig.ErrNotFound) {
		return Principal{}, err
	}

	var result Principal
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := runtimeconfig.LockAgentLabels(ctx, tx, initialLabels); err != nil {
			return err
		}
		now := s.now().UTC()
		candidate := Principal{
			RuntimeAgentID: runtimeAgentID, Labels: append([]string{}, initialLabels...),
			LabelRevision: 1, CreatedBy: "runtime-registration", CreatedAt: now,
			UpdatedBy: "runtime-registration", UpdatedAt: now,
		}
		repository := NewRepository(tx)
		inserted, err := repository.Insert(ctx, candidate)
		if err != nil {
			return err
		}
		if inserted {
			result = candidate
			return nil
		}
		result, err = repository.Get(ctx, runtimeAgentID)
		return err
	})
	return result, err
}

func (s *Service) Get(ctx context.Context, runtimeAgentID string) (Principal, error) {
	return s.repository.Get(ctx, runtimeAgentID)
}

// ListWithRequiredRuntimeAdapters reads one visible page and its distinct
// RuntimeConfig dependencies from the same database snapshot. The look-ahead
// row determines hasMore but never triggers adapter resolution.
func (s *Service) ListWithRequiredRuntimeAdapters(
	ctx context.Context, afterRuntimeAgentID string, pageLimit int,
) ([]AdapterRow, bool, error) {
	if !pagelimit.ValidPage(pageLimit) {
		return nil, false, invalid("Runtime Agent principal page is invalid")
	}
	var rows []AdapterRow
	var hasMore bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{
		IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly,
	}, func(tx pgx.Tx) error {
		var readErr error
		rows, hasMore, readErr = listAdapterPage(ctx, tx, afterRuntimeAgentID, pageLimit)
		return readErr
	})
	return rows, hasMore, err
}

func listAdapterPage(
	ctx context.Context, db persistencepostgres.DBTX, afterRuntimeAgentID string, pageLimit int,
) ([]AdapterRow, bool, error) {
	if !pagelimit.ValidPage(pageLimit) {
		return nil, false, invalid("Runtime Agent principal page is invalid")
	}
	principals, err := NewRepository(db).List(ctx, afterRuntimeAgentID, pagelimit.FetchRows(pageLimit))
	if err != nil {
		return nil, false, err
	}
	hasMore := len(principals) > pageLimit
	if hasMore {
		principals = principals[:pageLimit]
	}
	labels := make(map[string]struct{})
	for _, principal := range principals {
		for _, label := range principal.Labels {
			labels[label] = struct{}{}
		}
	}
	orderedLabels := slices.Sorted(maps.Keys(labels))
	repository := runtimeconfig.NewRepository(db)
	bindings, err := repository.GetBindingsByLabels(ctx, orderedLabels)
	if err != nil {
		return nil, false, err
	}
	refs := make(map[runtimeconfig.Ref]struct{}, len(bindings))
	for _, binding := range bindings {
		refs[binding.Ref] = struct{}{}
	}
	orderedRefs := make([]runtimeconfig.Ref, 0, len(refs))
	for ref := range refs {
		orderedRefs = append(orderedRefs, ref)
	}
	sort.Slice(orderedRefs, func(i, j int) bool { return orderedRefs[i].String() < orderedRefs[j].String() })
	versions, err := repository.GetVersionsByRefs(ctx, orderedRefs)
	if err != nil {
		return nil, false, err
	}
	rows := make([]AdapterRow, 0, len(principals))
	for _, principal := range principals {
		seen := make(map[string]struct{}, 3)
		for _, label := range principal.Labels {
			appendRequiredRuntimeAdapters(seen, versions[bindings[label].Ref])
		}
		rows = append(rows, AdapterRow{
			Principal: principal, RequiredRuntimeAdapters: sortedAdapters(seen),
		})
	}
	return rows, hasMore, nil
}

func (s *Service) RequiredRuntimeAdapters(ctx context.Context, labels []string) ([]string, error) {
	if err := runtimeconfig.ValidateAgentLabels(labels); err != nil {
		return nil, err
	}
	seen := make(map[string]struct{}, 2)
	repository := runtimeconfig.NewRepository(s.pool)
	for _, label := range labels {
		binding, err := repository.GetBinding(ctx, label)
		if err != nil {
			return nil, err
		}
		version, err := repository.GetVersionByRef(ctx, binding.Ref)
		if err != nil {
			return nil, err
		}
		appendRequiredRuntimeAdapters(seen, version)
	}
	return sortedAdapters(seen), nil
}

func appendRequiredRuntimeAdapters(seen map[string]struct{}, version runtimeconfig.Version) {
	if version.Spec.Worker.Telemetry.Present && !version.Spec.Worker.Telemetry.Clear {
		seen["otlp-http@1"] = struct{}{}
	}
	if version.Spec.Worker.HTTPProxy.Present && !version.Spec.Worker.HTTPProxy.Clear {
		seen["http-proxy@1"] = struct{}{}
	}
	if version.Spec.Worker.Caido.Present && !version.Spec.Worker.Caido.Clear {
		seen["caido-graphql@1"] = struct{}{}
	}
	if version.Spec.Worker.Interactsh.Present && !version.Spec.Worker.Interactsh.Clear {
		seen["interactsh-http@1"] = struct{}{}
	}
}

// sortedAdapters is never nil, so a principal without adapters lists none.
func sortedAdapters(seen map[string]struct{}) []string {
	result := make([]string, 0, len(seen))
	for adapter := range seen {
		result = append(result, adapter)
	}
	sort.Strings(result)
	return result
}

func (s *Service) ReplaceLabelsIdempotent(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	idempotencyKey string,
	actor string,
	at time.Time,
) (MutationResult, error) {
	receipt, err := prepareMutation(labelsReplaceOperation, runtimeAgentID, expectedRevision, labels, idempotencyKey)
	if err != nil {
		return MutationResult{}, err
	}
	if !runtimeconfig.ValidActor(actor) || at.IsZero() {
		return MutationResult{}, invalid("Runtime Agent label mutation is invalid")
	}
	if replay, found, replayErr := readReplay(ctx, s.pool, receipt); replayErr != nil || found {
		return replay, replayErr
	}
	optimistic, err := s.repository.Get(ctx, runtimeAgentID)
	if err != nil {
		return MutationResult{}, err
	}
	lockLabels := slices.Sorted(maps.Keys(setOf(optimistic.Labels, labels)))
	var result MutationResult
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var err error
		result, err = keyedMutation(ctx, tx, receipt, func() (MutationResult, error) {
			if _, err := runtimeconfig.NewRepository(tx).LockBindings(ctx, lockLabels); err != nil {
				if errors.Is(err, runtimeconfig.ErrNotFound) {
					return MutationResult{}, runtimeconfig.ErrUnknownLabel
				}
				return MutationResult{}, err
			}
			repository := NewRepository(tx)
			locked, err := repository.Lock(ctx, runtimeAgentID)
			if err != nil {
				return MutationResult{}, err
			}
			if locked.LabelRevision != expectedRevision ||
				locked.LabelRevision != optimistic.LabelRevision || !slices.Equal(locked.Labels, optimistic.Labels) {
				return MutationResult{}, runtimeconfig.ErrPrecondition
			}
			if err := runtimeconfig.ValidateLockedAgentLabels(ctx, tx, labels); err != nil {
				return MutationResult{}, err
			}
			updated, err := repository.ReplaceLabels(ctx, runtimeAgentID, expectedRevision, labels, actor, at)
			if err != nil {
				return MutationResult{}, err
			}
			return MutationResult{Principal: &updated}, nil
		})
		return err
	})
	return result, err
}

func (s *Service) DeleteIdempotent(
	ctx context.Context,
	runtimeAgentID string,
	expectedRevision uint64,
	idempotencyKey string,
	actor string,
	at time.Time,
) (MutationResult, error) {
	receipt, err := prepareMutation(deleteOperation, runtimeAgentID, expectedRevision, []string{}, idempotencyKey)
	if err != nil {
		return MutationResult{}, err
	}
	if !runtimeconfig.ValidActor(actor) || at.IsZero() {
		return MutationResult{}, invalid("Runtime Agent principal delete is invalid")
	}
	if replay, found, replayErr := readReplay(ctx, s.pool, receipt); replayErr != nil || found {
		return replay, replayErr
	}
	if s.deletionGuard == nil {
		return MutationResult{}, errors.New("Runtime Agent principal deletion guard is not configured")
	}
	release, err := s.deletionGuard.BeginPrincipalDeletion(runtimeAgentID)
	if err != nil {
		return MutationResult{}, err
	}
	defer release()
	var result MutationResult
	err = persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var err error
		result, err = keyedMutation(ctx, tx, receipt, func() (MutationResult, error) {
			repository := NewRepository(tx)
			principal, err := repository.Lock(ctx, runtimeAgentID)
			if err != nil {
				return MutationResult{}, err
			}
			if principal.LabelRevision != expectedRevision {
				return MutationResult{}, runtimeconfig.ErrPrecondition
			}
			if len(principal.Labels) != 0 {
				return MutationResult{}, ErrInUse
			}
			inUse, err := repository.HasLiveAllocation(ctx, runtimeAgentID)
			if err != nil {
				return MutationResult{}, err
			}
			if inUse {
				return MutationResult{}, ErrInUse
			}
			if err := repository.Delete(ctx, runtimeAgentID, expectedRevision); err != nil {
				return MutationResult{}, err
			}
			return MutationResult{Deleted: true}, nil
		})
		return err
	})
	return result, err
}

func prepareMutation(
	kind, runtimeAgentID string,
	expectedRevision uint64,
	labels []string,
	idempotencyKey string,
) (runtimeconfig.ManagementReceipt, error) {
	if (kind != labelsReplaceOperation && kind != deleteOperation) ||
		validateRuntimeAgentID(runtimeAgentID) != nil || expectedRevision == 0 ||
		runtimeconfig.ValidateAgentLabels(labels) != nil || kind == deleteOperation && len(labels) != 0 {
		return runtimeconfig.ManagementReceipt{}, invalid("Runtime Agent principal mutation is invalid")
	}
	keyDigest, err := runtimeconfig.DigestIdempotencyKey(idempotencyKey)
	if err != nil {
		return runtimeconfig.ManagementReceipt{}, err
	}
	request, err := json.Marshal(struct {
		Kind             string   `json:"kind"`
		RuntimeAgentID   string   `json:"runtimeAgentId"`
		ExpectedRevision uint64   `json:"expectedRevision"`
		Labels           []string `json:"labels"`
	}{kind, runtimeAgentID, expectedRevision, labels})
	if err != nil {
		return runtimeconfig.ManagementReceipt{}, invalid("Runtime Agent principal mutation cannot be normalized")
	}
	return runtimeconfig.ManagementReceipt{
		KeyDigest: keyDigest, RequestDigest: contentdigest.Bytes(request), Kind: kind, ResourceID: runtimeAgentID,
	}, nil
}

func readReplay(
	ctx context.Context, db persistencepostgres.DBTX, receipt runtimeconfig.ManagementReceipt,
) (MutationResult, bool, error) {
	var wire mutationResultWire
	found, err := runtimeconfig.ReadManagementReplay(ctx, db, receipt, &wire)
	return replayed(wire, found, err)
}

func replayed(wire mutationResultWire, found bool, err error) (MutationResult, bool, error) {
	if err != nil || !found {
		return MutationResult{}, false, err
	}
	if (wire.Principal == nil) == !wire.Deleted {
		return MutationResult{}, false, errors.New("invalid Runtime Agent management replay")
	}
	return MutationResult{Principal: wire.Principal, Deleted: wire.Deleted, Replayed: true}, true, nil
}

// keyedMutation runs one Runtime Agent principal mutation in tx as a Runtime
// management keyed mutation (PER-R24): a recorded outcome under the key
// replays without running mutate. The actor and time are recorded on the
// principal, not in the receipt.
func keyedMutation(
	ctx context.Context, tx pgx.Tx, receipt runtimeconfig.ManagementReceipt,
	mutate func() (MutationResult, error),
) (MutationResult, error) {
	var wire mutationResultWire
	var changed MutationResult
	replay, err := runtimeconfig.KeyedManagementMutation(ctx, tx, receipt, &wire, func() (any, error) {
		var err error
		if changed, err = mutate(); err != nil {
			return nil, err
		}
		if (changed.Principal == nil) == !changed.Deleted {
			return nil, invalid("Runtime Agent management audit is invalid")
		}
		return mutationResultWire{Principal: changed.Principal, Deleted: changed.Deleted}, nil
	})
	if err != nil || !replay {
		return changed, err
	}
	result, _, err := replayed(wire, true, nil)
	return result, err
}

func setOf(groups ...[]string) map[string]struct{} {
	result := make(map[string]struct{})
	for _, group := range groups {
		for _, value := range group {
			result[value] = struct{}{}
		}
	}
	return result
}
