package evalservice

import (
	"context"
	"encoding/json"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type ExperimentView struct {
	ID                     string                   `json:"experimentId"`
	PortableID             string                   `json:"portableExperimentId"`
	ProjectID              string                   `json:"projectId"`
	Name                   string                   `json:"name"`
	ControlMode            evaldomain.ControlMode   `json:"controlMode"`
	ExecutionKind          string                   `json:"executionKind"`
	State                  evaldomain.State         `json:"state"`
	Revision               int64                    `json:"revision"`
	PlanSHA256             *string                  `json:"planSha256"`
	ViewSnapshot           *string                  `json:"viewSnapshot"`
	Summary                json.RawMessage          `json:"summary"`
	UpdatedAt              time.Time                `json:"updatedAt"`
	LastProducerActivityAt *time.Time               `json:"lastProducerActivityAt"`
	Draft                  json.RawMessage          `json:"draft,omitempty"`
	Setup                  json.RawMessage          `json:"setup,omitempty"`
	Expected               int                      `json:"expectedMembers"`
	AllowedCommands        []evaldomain.CommandKind `json:"allowedCommands"`
	Diagnostics            []json.RawMessage        `json:"diagnostics"`
	StartedAt              *time.Time               `json:"startedAt"`
	DeadlineAt             *time.Time               `json:"deadlineAt"`
	ObservedTokens         int64                    `json:"observedTokens"`
	DeletionRequestedAt    *time.Time               `json:"deletionRequestedAt"`
}

func (s *Service) Get(ctx context.Context, owner, id string) (ExperimentView, error) {
	var out ExperimentView
	err := pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		e, err := st.Get(ctx, owner, id)
		if err != nil {
			return err
		}
		out = ExperimentView{
			ID:                     e.ID,
			PortableID:             e.PortableID,
			ProjectID:              e.ProjectID,
			Name:                   e.Name,
			ControlMode:            e.ControlMode,
			State:                  e.State,
			Revision:               e.Revision,
			UpdatedAt:              e.UpdatedAt,
			LastProducerActivityAt: e.LastProducerActivityAt,
			Expected:               e.Expected,
			AllowedCommands:        []evaldomain.CommandKind{},
			Diagnostics:            []json.RawMessage{},
			StartedAt:              e.StartedAt,
			DeadlineAt:             e.DeadlineAt,
			ObservedTokens:         e.ObservedTokens,
			DeletionRequestedAt:    e.DeletionRequestedAt,
		}
		if len(e.Diagnostic) > 0 {
			out.Diagnostics = append(out.Diagnostics, e.Diagnostic)
		}
		out.UpdatedAt = out.UpdatedAt.UTC()
		out.StartedAt, out.DeadlineAt = utcTime(out.StartedAt), utcTime(out.DeadlineAt)
		out.LastProducerActivityAt, out.DeletionRequestedAt = utcTime(out.LastProducerActivityAt), utcTime(out.DeletionRequestedAt)
		if e.State == "draft" || e.State == "preparing" {
			var draft evaldomain.Draft
			if err = json.Unmarshal(e.Draft.Bytes(), &draft); err != nil {
				return err
			}
			out.Draft = e.Draft.Bytes()
			out.ExecutionKind = draft.Variants[0].Kind
			out.Expected = len(draft.CaseIDs) * draft.Repetitions * 2
		} else {
			plan, err := st.FrozenPlan(ctx, owner, id)
			if notFound(err) && e.Draft.Kind() == "Draft" {
				// Deleting an unprepared draft still needs a safe authoring projection.
				var draft evaldomain.Draft
				if err = json.Unmarshal(e.Draft.Bytes(), &draft); err != nil {
					return err
				}
				out.ExecutionKind = draft.Variants[0].Kind
				out.Expected = len(draft.CaseIDs) * draft.Repetitions * 2
				out.Setup, err = jsonBytes(draft.Setup())
				if err != nil {
					return err
				}
			} else {
				if err != nil {
					return err
				}
				out.Setup = plan.Setup
				out.PlanSHA256 = &plan.SHA256
				var setup struct {
					Variants []evaldomain.Variant `json:"variants"`
				}
				if err = json.Unmarshal(plan.Setup, &setup); err != nil {
					return err
				}
				out.ExecutionKind = setup.Variants[0].Kind
			}
		}
		out.AllowedCommands = e.Lifecycle().AllowedCommands()
		raw, err := jsonBytes(out)
		if err != nil {
			return err
		}
		return evaldomain.Validate("Experiment", raw)
	})
	return out, err
}

func utcTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	utc := value.UTC()
	return &utc
}

func (s *Service) List(ctx context.Context, p evalstore.SummaryPageParams) (evalstore.SummaryPage, error) {
	return evalstore.NewPostgresStore(s.pool).SummaryPage(ctx, p)
}
func (s *Service) Datasets(ctx context.Context, scope evalstore.Scope, afterID, afterRevision string, limit int, revision *int64) (evalstore.DatasetPage, error) {
	return evalstore.NewPostgresStore(s.pool).DatasetPage(ctx, scope, afterID, afterRevision, limit, revision)
}
func (s *Service) Dataset(ctx context.Context, scope evalstore.Scope, id, revision string) (evalstore.DatasetRevision, error) {
	return evalstore.NewPostgresStore(s.pool).Dataset(ctx, scope, id, revision)
}

func (s *Service) PutDataset(ctx context.Context, scope evalstore.Scope, doc evaldomain.Frozen, m evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	var data evaldomain.DatasetInput
	if err := evaldomain.DecodeInto("DatasetInput", doc.Bytes(), &data); err != nil {
		return evalstore.Receipt{}, err
	}
	replay, err := evalstore.NewPostgresStore(s.pool).Replay(ctx, scope, "datasets", "dataset-create", m)
	if err != nil {
		return evalstore.Receipt{}, err
	}
	if replay != nil {
		return *replay, nil
	}
	revision, err := newID("revision-")
	if err != nil {
		return evalstore.Receipt{}, err
	}
	var result evalstore.Receipt
	err = pg.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		for _, c := range data.Cases {
			for _, ref := range c.Inputs {
				if err := verifyArtifact(ctx, tx, artifactService, scope.OwnerID, ref); err != nil {
					return err
				}
			}
		}
		var err error
		result, err = evalstore.NewTxStore(tx).PutDataset(ctx, scope, revision, doc, m)
		return err
	})
	return result, err
}

func (s *Service) ScopeForMutation(ctx context.Context, owner, id, operation, key string) (evalstore.Scope, error) {
	return evalstore.NewPostgresStore(s.pool).ScopeForMutation(ctx, owner, id, operation, key)
}
func (s *Service) UpdateDraft(ctx context.Context, scope evalstore.Scope, id string, doc evaldomain.Frozen, m evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	var result evalstore.Receipt
	err := s.tx(ctx, func(st *evalstore.Store) error {
		var err error
		result, err = st.UpdateDraft(ctx, scope, id, doc, m)
		return err
	})
	return result, err
}
func (s *Service) Delete(ctx context.Context, scope evalstore.Scope, id string, m evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	var result evalstore.Receipt
	err := s.tx(ctx, func(st *evalstore.Store) error {
		var err error
		result, err = st.BeginDeletion(ctx, scope, id, m)
		return err
	})
	return result, err
}
func (s *Service) GetCommand(ctx context.Context, owner, id, command string) (evalstore.CommandRecord, *string, error) {
	st := evalstore.NewPostgresStore(s.pool)
	record, err := st.CommandRecord(ctx, owner, id, command)
	if err != nil {
		return record, nil, err
	}
	plan, err := st.FrozenPlan(ctx, owner, id)
	if notFound(err) {
		return record, nil, nil
	}
	if err != nil {
		return record, nil, err
	}
	return record, &plan.SHA256, nil
}
