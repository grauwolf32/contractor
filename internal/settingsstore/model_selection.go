package settingsstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"regexp"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

var modelIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]{0,74}$`)
var ErrModelSelectionInvalid = apierror.New(apierror.InvalidRequest, "Model selection is invalid")

// ModelSelectionValue is the complete default plus optional role overrides.
// It contains model profile IDs only, never provider parameters or credentials.
type ModelSelectionValue struct {
	DefaultModel string            `json:"defaultModel" strictjson:"required"`
	Roles        map[string]string `json:"roles" strictjson:"required"`
}

func (v ModelSelectionValue) Validate() error {
	if !modelIDPattern.MatchString(v.DefaultModel) || v.Roles == nil || len(v.Roles) > 4 {
		return ErrModelSelectionInvalid
	}
	for role, model := range v.Roles {
		switch role {
		case "planner", "worker", "summarizer", "audit_completion_worker":
		default:
			return ErrModelSelectionInvalid
		}
		if !modelIDPattern.MatchString(model) {
			return ErrModelSelectionInvalid
		}
	}
	return nil
}

func (v ModelSelectionValue) Clone() ModelSelectionValue {
	return ModelSelectionValue{DefaultModel: v.DefaultModel, Roles: maps.Clone(v.Roles)}
}

// ModelSelectionSettings is the durable singleton read under a revision.
type ModelSelectionSettings struct {
	Value     ModelSelectionValue
	Revision  uint64
	UpdatedAt time.Time
}

type modelSelectionTransactionKey struct{}

// WithModelSelectionTransaction binds model-default reads to an existing
// preflight transaction, avoiding a nested pool acquisition and mixed snapshots.
func WithModelSelectionTransaction(ctx context.Context, tx pgx.Tx) context.Context {
	return context.WithValue(ctx, modelSelectionTransactionKey{}, tx)
}

func (s *PostgresStore) modelSelectionDB(ctx context.Context) pg.DBTX {
	if tx, ok := ctx.Value(modelSelectionTransactionKey{}).(pgx.Tx); ok {
		return tx
	}
	return s.db
}

// InitializeModelSelection seeds an absent row only; restart cannot replace
// an operator's saved selection with the deployment bootstrap value.
func (s *PostgresStore) InitializeModelSelection(ctx context.Context, value ModelSelectionValue) (ModelSelectionSettings, error) {
	if err := value.Validate(); err != nil {
		return ModelSelectionSettings{}, err
	}
	if s == nil || s.db == nil {
		return ModelSelectionSettings{}, ErrInvariant
	}
	roles, err := json.Marshal(value.Roles)
	if err != nil {
		return ModelSelectionSettings{}, err
	}
	if _, err = s.modelSelectionDB(ctx).Exec(ctx, `INSERT INTO model_selection_settings(singleton, default_model, role_models) VALUES(true,$1,$2::jsonb) ON CONFLICT(singleton) DO NOTHING`, value.DefaultModel, string(roles)); err != nil {
		return ModelSelectionSettings{}, pg.WrapError("initialize model selection", err)
	}
	return s.GetModelSelection(ctx)
}

func (s *PostgresStore) GetModelSelection(ctx context.Context) (ModelSelectionSettings, error) {
	if s == nil || s.db == nil {
		return ModelSelectionSettings{}, ErrInvariant
	}
	value, err := scanModelSelection(s.modelSelectionDB(ctx).QueryRow(ctx, `SELECT default_model, role_models, revision::text, updated_at FROM model_selection_settings WHERE singleton=true`))
	if errors.Is(err, pgx.ErrNoRows) {
		return ModelSelectionSettings{}, fmt.Errorf("%w: model selection is not initialized", ErrInvariant)
	}
	if err != nil {
		return ModelSelectionSettings{}, pg.WrapError("read model selection", err)
	}
	return value, nil
}

func (s *PostgresStore) UpdateModelSelection(ctx context.Context, value ModelSelectionValue, expected uint64) (ModelSelectionSettings, error) {
	if err := value.Validate(); err != nil {
		return ModelSelectionSettings{}, err
	}
	if expected == 0 || expected == math.MaxUint64 {
		return ModelSelectionSettings{}, ErrModelSelectionInvalid
	}
	if s == nil || s.db == nil {
		return ModelSelectionSettings{}, ErrInvariant
	}
	roles, err := json.Marshal(value.Roles)
	if err != nil {
		return ModelSelectionSettings{}, err
	}
	result, err := scanModelSelection(s.modelSelectionDB(ctx).QueryRow(ctx, updateModelSelectionSQL, value.DefaultModel, string(roles), strconv.FormatUint(expected, 10)))
	if err == nil {
		return result, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return ModelSelectionSettings{}, pg.WrapError("replace model selection", err)
	}
	current, getErr := s.GetModelSelection(ctx)
	if getErr != nil {
		return ModelSelectionSettings{}, getErr
	}
	if current.Revision != expected || current.Value.DefaultModel != value.DefaultModel || !maps.Equal(current.Value.Roles, value.Roles) {
		return ModelSelectionSettings{}, ErrPrecondition
	}
	return ModelSelectionSettings{}, fmt.Errorf("%w: model selection exact update returned no row", ErrInvariant)
}

func scanModelSelection(row rowScanner) (ModelSelectionSettings, error) {
	var result ModelSelectionSettings
	var roles []byte
	var revision string
	if err := row.Scan(&result.Value.DefaultModel, &roles, &revision, &result.UpdatedAt); err != nil {
		return ModelSelectionSettings{}, err
	}
	if err := strictjson.Decode(roles, &result.Value.Roles); err != nil {
		return ModelSelectionSettings{}, fmt.Errorf("%w: invalid model selection roles", ErrInvariant)
	}
	var err error
	result.Revision, err = strconv.ParseUint(revision, 10, 64)
	if err != nil || result.Revision == 0 || result.UpdatedAt.IsZero() || result.Value.Validate() != nil {
		return ModelSelectionSettings{}, ErrInvariant
	}
	result.UpdatedAt = result.UpdatedAt.UTC()
	return result, nil
}
