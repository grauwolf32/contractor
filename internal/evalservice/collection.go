package evalservice

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

//go:embed output_collection.go
var outputNormalizerSource []byte

//go:embed usage_collection.go
var usageNormalizerSource []byte

func NormalizerSHA256() string {
	return evaldomain.Digest(append(append([]byte{}, outputNormalizerSource...), usageNormalizerSource...))
}

type preparedSetup struct {
	Variants   []evaldomain.Variant  `json:"variants"`
	Checks     []evaldomain.Check    `json:"checks"`
	Comparison evaldomain.Comparison `json:"comparison"`
}

// CollectView observes a bounded dirty-member batch; it never admits an
// execution or calls a producer. Terminal experiments remain claimable while
// this private projection queue is dirty.
func (s *Service) CollectView(ctx context.Context, e evalstore.Experiment, claim evalstore.Claim) error {
	if e.Expected == 0 || e.DeletionRequestedAt != nil {
		return nil
	}
	reader := evalstore.NewPostgresStore(s.pool)
	dirty, err := reader.DirtyMembers(ctx, e.OwnerID, e.ID, evaldomain.CollectionBatchSize)
	if err != nil {
		return err
	}
	var faults []error
	for _, member := range dirty {
		err = s.collectMember(ctx, e, member.MemberID, claim)
		if err != nil {
			faults = append(faults, err)
			if ctx.Err() == nil {
				faults = append(faults, reader.RotateProjectionFailure(ctx, e.OwnerID, e.ID, member.MemberID))
			}
		}
	}
	if len(faults) > 0 {
		return errors.Join(faults...)
	}
	return pg.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		plan, err := st.FrozenPlan(ctx, e.OwnerID, e.ID)
		if err != nil {
			return err
		}
		var setup preparedSetup
		if err = json.Unmarshal(plan.Setup, &setup); err != nil {
			return err
		}
		verified, err := verifiedComparisonPins(ctx, st, e, plan, setup.Comparison)
		if err != nil {
			return err
		}
		_, err = st.PublishView(ctx, scope(e), e.ID, claim, setup.Comparison, verified)
		var domainError *evaldomain.Error
		if errors.As(err, &domainError) && domainError.Code == "eval_not_ready" {
			return nil
		}
		return err
	})
}
func (s *Service) collectMember(ctx context.Context, e evalstore.Experiment, member string, claim evalstore.Claim) error {
	return pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		locked, err := st.LockCollection(ctx, scope(e), e.ID, claim)
		if err != nil {
			return err
		}
		e = locked
		m, err := st.Member(ctx, e.OwnerID, e.ID, member)
		if err != nil {
			return err
		}
		row, err := st.ExecutionObservation(ctx, e.OwnerID, e.ID, member)
		if err != nil {
			return err
		}
		view := executionMember(row)
		inventory, err := st.Inventory(ctx, e.OwnerID, e.ID, member)
		if err != nil {
			return err
		}
		plan, err := st.FrozenPlan(ctx, e.OwnerID, e.ID)
		if err != nil {
			return err
		}
		var setup preparedSetup
		if err = json.Unmarshal(plan.Setup, &setup); err != nil {
			return err
		}
		var observation *evaldomain.ResultInput
		if view.Execution.Ref != nil {
			usage, err := observedUsage(ctx, tx, e.OwnerID, member, *view.Execution, inventory)
			if err != nil {
				return err
			}
			outputs, err := collectOutputs(ctx, tx, e.OwnerID, m, *view.Execution, inventory)
			if err != nil {
				return err
			}
			revision := NormalizerSHA256()
			observation = &evaldomain.ResultInput{
				SchemaVersion: evaldomain.ResultSchemaVersion,
				PlanSHA256:    plan.SHA256,
				MemberID:      member,
				Source:        evaldomain.Source{System: "contractor", ID: evaldomain.NativeCollectorSource, Revision: &revision},
				Execution:     *view.Execution,
				Usage:         usage,
				Collection:    outputs.Collection,
				Outputs:       outputs.Outputs,
				Evidence:      outputs.Evidence,
			}
			view.Usage = &usage
			if e.ControlMode == "server" && isTerminal(view.Execution.State) {
				if err = s.collectNativeRecords(ctx, st, tx, e, m, claim, setup.Checks, *observation); err != nil {
					return err
				}
			}
		}
		complete, err := applySelectedRecords(ctx, st, e, m, setup.Checks, observation, &view)
		if err != nil {
			return err
		}
		revision, err := st.ProjectionRevision(ctx, e.OwnerID, e.ID, member)
		if err != nil {
			return err
		}
		var observed json.RawMessage
		if observation != nil {
			observed, err = jsonBytes(observation)
			if err != nil {
				return err
			}
		}
		return st.ProjectMember(ctx, scope(e), e.ID, member, claim, revision, view, complete, observed)
	})
}
