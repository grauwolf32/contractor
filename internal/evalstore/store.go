// Package evalstore persists managed eval authority. Multi-statement mutations
// require a caller-owned transaction, allowing submission and lifecycle services
// to compose stores without hidden commits. It never dispatches an execution.
package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var (
	ErrTransaction = errors.New("evalstore requires a caller-owned transaction")
	ErrClaimLost   = errors.New("evalstore controller claim is stale")
	ErrDrain       = errors.New("evalstore has unresolved operations or execution dependencies")
)

type Store struct {
	db pg.DBTX
	tx pgx.Tx
}

func NewPostgresStore(db pg.DBTX) *Store { return &Store{db: db} }

func NewTxStore(tx pgx.Tx) *Store { return &Store{db: tx, tx: tx} }

func (s *Store) requireTx() error {
	if s.tx == nil {
		return ErrTransaction
	}
	return nil
}

type Scope struct{ OwnerID, ProjectID string }

type Receipt struct {
	Response json.RawMessage
	Replayed bool
}

type Experiment struct {
	ID, OwnerID, ProjectID, PortableID, Name                           string
	State                                                              evaldomain.State
	ControlMode                                                        evaldomain.ControlMode
	Revision                                                           int64
	DatasetID, DatasetRevision                                         *string
	Draft                                                              evaldomain.Frozen `json:"-"`
	Expected, Outstanding, MaxInFlight                                 int
	WallMS                                                             int64
	TokenLimit                                                         *int64
	ObservedTokens                                                     int64
	StartedAt, DeadlineAt, LastProducerActivityAt, DeletionRequestedAt *time.Time
	Diagnostic                                                         json.RawMessage
	ViewGeneration                                                     int64
	CreatedAt, UpdatedAt                                               time.Time
}

func (e Experiment) Lifecycle() evaldomain.Lifecycle {
	return evaldomain.Lifecycle{
		State: e.State, ControlMode: e.ControlMode, Outstanding: e.Outstanding,
		HasPlan: e.Expected > 0, HasDraft: e.Draft.Kind() == "Draft",
		DeletionRequested: e.DeletionRequestedAt != nil,
	}
}

const experimentColumns = `experiment_id,owner_id,project_id,portable_id,control_mode,name,state,revision,draft,dataset_id,dataset_revision,expected_count,outstanding_count,max_in_flight,wall_ms,token_limit,observed_tokens,started_at,deadline_at,last_producer_activity_at,deletion_requested_at,diagnostic,view_generation,created_at,updated_at`

type scanner interface{ Scan(...any) error }

func scanExperiment(row scanner) (Experiment, error) {
	var e Experiment
	var draft []byte
	err := row.Scan(&e.ID, &e.OwnerID, &e.ProjectID, &e.PortableID, &e.ControlMode, &e.Name, &e.State, &e.Revision, &draft, &e.DatasetID, &e.DatasetRevision, &e.Expected, &e.Outstanding, &e.MaxInFlight, &e.WallMS, &e.TokenLimit, &e.ObservedTokens, &e.StartedAt, &e.DeadlineAt, &e.LastProducerActivityAt, &e.DeletionRequestedAt, &e.Diagnostic, &e.ViewGeneration, &e.CreatedAt, &e.UpdatedAt)
	if err != nil {
		return e, normalize(err)
	}
	if draft != nil {
		e.Draft, err = evaldomain.Freeze("Draft", draft)
	}
	return e, err
}

func normalize(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return evaldomain.Failure("eval_not_found")
	}
	switch pg.Classify(err) {
	case pg.ClassDuplicate:
		return evaldomain.Failure("eval_member_conflict")
	case pg.ClassReferenceViolation:
		return evaldomain.Failure("eval_not_ready")
	case pg.ClassDeleting:
		return evaldomain.Failure("eval_project_deleting")
	}
	return err
}

func validScope(scope Scope) bool {
	return scope.OwnerID != "" && len(scope.OwnerID) <= 256 && contracts.ValidResourceID(scope.ProjectID)
}

func (s *Store) project(ctx context.Context, scope Scope, lock bool) (bool, error) {
	if !validScope(scope) {
		return false, evaldomain.Failure("eval_invalid")
	}
	query := `SELECT lifecycle_state FROM projects WHERE owner_id=$1 AND project_id=$2 AND kind='evaluation'`
	if lock {
		if err := s.requireTx(); err != nil {
			return false, err
		}
		query += ` FOR SHARE`
	}
	var state string
	err := s.db.QueryRow(ctx, query, scope.OwnerID, scope.ProjectID).Scan(&state)
	return state == "active", normalize(err)
}

// mutateJSON runs one managed Eval mutation as a keyed mutation (PER-R23):
// a key is unique within its owner, Project, resource and operation. Replay
// precedes both revision and deletion rejection, but ownership is always
// verified first. The advisory lock does not refresh a REPEATABLE READ
// snapshot taken before it was granted; there, a receipt committed after the
// snapshot fails the claim as a serialization failure, which the caller
// retries into a replay, rather than as a member conflict.
func (s *Store) mutateJSON(ctx context.Context, scope Scope, resource, operation string, id evaldomain.MutationIdentity, fn func() (json.RawMessage, error)) (Receipt, error) {
	if err := s.requireTx(); err != nil {
		return Receipt{}, err
	}
	if resource == "" || operation == "" || id.Key == "" || len(id.Key) > 128 || !contentdigest.Valid(id.RequestSHA256) {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	active, err := s.project(ctx, scope, true)
	if err != nil {
		return Receipt{}, err
	}
	key := pg.EvalReceiptKey(scope.OwnerID, scope.ProjectID, resource, operation, id.Key)
	var mutationErr error
	receipt, replayed, err := pg.KeyedMutation(ctx, s.tx, key, id.RequestSHA256, func() (pg.Receipt, error) {
		if !active {
			mutationErr = evaldomain.Failure("eval_project_deleting")
			return pg.Receipt{}, mutationErr
		}
		response, err := fn()
		if err != nil {
			mutationErr = err
			return pg.Receipt{}, err
		}
		return pg.Receipt{
			Operation: operation, ResourceID: resource, Response: response,
			OwnerID: scope.OwnerID, ProjectID: scope.ProjectID,
		}, nil
	})
	switch {
	case mutationErr != nil:
		return Receipt{}, mutationErr
	case errors.Is(err, pg.ErrReceiptMismatch):
		return Receipt{}, evaldomain.Failure("eval_idempotency_conflict")
	case errors.Is(err, pg.ErrReceiptClaimed):
		return Receipt{}, evaldomain.Failure("eval_member_conflict")
	case err != nil:
		return Receipt{}, normalize(err)
	}
	return Receipt{Response: receipt.Response, Replayed: replayed}, nil
}

func (s *Store) Get(ctx context.Context, owner, id string) (Experiment, error) {
	return scanExperiment(s.db.QueryRow(ctx, `SELECT `+experimentColumns+` FROM eval_experiments WHERE owner_id=$1 AND experiment_id=$2`, owner, id))
}

func (s *Store) locked(ctx context.Context, scope Scope, id string) (Experiment, error) {
	if err := s.requireTx(); err != nil {
		return Experiment{}, err
	}
	return scanExperiment(s.db.QueryRow(ctx, `SELECT `+experimentColumns+` FROM eval_experiments WHERE owner_id=$1 AND project_id=$2 AND experiment_id=$3 FOR UPDATE`, scope.OwnerID, scope.ProjectID, id))
}

func checkMutable(e Experiment, id evaldomain.MutationIdentity) error {
	if e.DeletionRequestedAt != nil {
		return evaldomain.Failure("eval_project_deleting")
	}
	if id.ExpectedRevision == nil {
		return evaldomain.Failure("eval_precondition_required")
	}
	_, err := evaldomain.CheckMutation(id, nil, uint64(e.Revision))
	return err
}

const observe = `updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')`
const advance = `revision=revision+1,` + observe

func bytesOf(value any) []byte {
	b, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("evalstore internal JSON type: %T", value))
	}
	return b
}
