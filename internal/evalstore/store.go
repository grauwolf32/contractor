// Package evalstore persists managed eval authority. Multi-statement mutations
// require a caller-owned transaction, allowing submission and lifecycle services
// to compose stores without hidden commits. It never dispatches an execution.
package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

var (
	ErrTransaction = errors.New("evalstore requires a caller-owned transaction")
	ErrClaimLost   = errors.New("evalstore controller claim is stale")
	ErrDrain       = errors.New("evalstore has unresolved operations or execution dependencies")
	resourceID     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)
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
	var p *pgconn.PgError
	if errors.As(err, &p) {
		switch p.Code {
		case "23505":
			return evaldomain.Failure("eval_member_conflict")
		case "23503":
			return evaldomain.Failure("eval_not_ready")
		case "55000":
			return evaldomain.Failure("eval_project_deleting")
		}
	}
	return err
}

func validScope(scope Scope) bool {
	return scope.OwnerID != "" && len(scope.OwnerID) <= 256 && resourceID.MatchString(scope.ProjectID)
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

// mutate serializes a key within its owner/resource scope. Replay precedes both
// revision and deletion rejection, but ownership is always verified first.
var sha256Pattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func (s *Store) mutateJSON(ctx context.Context, scope Scope, resource, operation string, id evaldomain.MutationIdentity, fn func() (json.RawMessage, error)) (Receipt, error) {
	if err := s.requireTx(); err != nil {
		return Receipt{}, err
	}
	if resource == "" || operation == "" || id.Key == "" || len(id.Key) > 128 || !sha256Pattern.MatchString(id.RequestSHA256) {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	active, err := s.project(ctx, scope, true)
	if err != nil {
		return Receipt{}, err
	}
	key, _ := json.Marshal([]string{scope.OwnerID, scope.ProjectID, resource, operation, id.Key})
	if _, err = s.db.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1,0))`, string(key)); err != nil {
		return Receipt{}, err
	}
	var digest string
	var response []byte
	err = s.db.QueryRow(ctx, `
SELECT request_sha256, response
FROM eval_mutation_receipts
WHERE owner_id=$1
    AND project_id=$2
    AND resource_id=$3
    AND operation=$4
    AND operation_key=$5
`, scope.OwnerID, scope.ProjectID, resource, operation, id.Key).Scan(&digest, &response)
	if err == nil {
		if digest != id.RequestSHA256 {
			return Receipt{}, evaldomain.Failure("eval_idempotency_conflict")
		}
		return Receipt{Response: response, Replayed: true}, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Receipt{}, err
	}
	if !active {
		return Receipt{}, evaldomain.Failure("eval_project_deleting")
	}
	response, err = fn()
	if err != nil {
		return Receipt{}, err
	}
	// The advisory lock does not refresh a REPEATABLE READ snapshot taken
	// before it was granted. There, DO NOTHING turns a receipt committed after
	// the snapshot into a serialization failure, which the caller retries into
	// a replay, rather than a unique violation reported as a member conflict.
	tag, err := s.db.Exec(ctx, `
INSERT INTO eval_mutation_receipts(owner_id, project_id, resource_id, operation, operation_key, request_sha256, expected_revision, response)
VALUES($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT DO NOTHING
`, scope.OwnerID, scope.ProjectID, resource, operation, id.Key, id.RequestSHA256, id.ExpectedRevision, response)
	if err != nil {
		return Receipt{}, normalize(err)
	}
	if tag.RowsAffected() != 1 {
		return Receipt{}, evaldomain.Failure("eval_member_conflict")
	}
	return Receipt{Response: response}, nil
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

const advance = `revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')`

func bytesOf(value any) []byte {
	b, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("evalstore internal JSON type: %T", value))
	}
	return b
}
