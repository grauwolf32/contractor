package evalservice

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type ReviewContext struct {
	MemberID     string                    `json:"memberId"`
	ResultSHA256 string                    `json:"resultSha256"`
	Checks       []evaldomain.PrivateCheck `json:"checks"`
	Evidence     []evaldomain.Evidence     `json:"evidence"`
	Gaps         []string                  `json:"gaps"`
	Result       evaldomain.ResultInput    `json:"result"`
	Policy       []evaldomain.Check        `json:"policy"`
	Revision     int64                     `json:"revision"`
}

func reviewRubric(ctx context.Context, st *evalstore.Store, e evalstore.Experiment, check evaldomain.Check) (evaldomain.PrivateCheck, error) {
	var rubric evaldomain.PrivateCheck
	if check.ImplementationSHA256 != HumanPolicySHA256() || check.RubricRevision == "" {
		return rubric, evaldomain.Failure("eval_pin_mismatch")
	}
	resource, err := st.PlanResource(ctx, e.OwnerID, e.ID, "private/"+check.ID+".json")
	if err != nil {
		return rubric, err
	}
	if err = evaldomain.DecodeInto("PrivateCheck", resource.Document.Bytes(), &rubric); err != nil {
		return rubric, err
	}
	if rubric.ID != check.ID || rubric.Revision != check.RubricRevision {
		return rubric, evaldomain.Failure("eval_pin_mismatch")
	}
	return rubric, nil
}

func (s *Service) Review(ctx context.Context, owner, id, member, resultDigest string) (ReviewContext, error) {
	out := ReviewContext{MemberID: member, Checks: []evaldomain.PrivateCheck{}, Evidence: []evaldomain.Evidence{}, Gaps: []string{}}
	err := pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		e, err := st.Get(ctx, owner, id)
		if err != nil {
			return err
		}
		out.Revision = e.Revision
		if resultDigest == "" {
			selected, err := st.Selected(ctx, owner, id, member)
			if err != nil {
				return err
			}
			if selected == nil {
				return evaldomain.Failure("eval_not_ready")
			}
			resultDigest = selected.ResultSHA256
		}
		record, err := st.Record(ctx, owner, id, member, evaldomain.RecordKindResult, resultDigest)
		if err != nil {
			return err
		}
		if err = json.Unmarshal(record.Document.Bytes(), &out.Result); err != nil {
			return err
		}
		out.ResultSHA256 = resultDigest
		out.Evidence = out.Result.Evidence
		out.Gaps = append(out.Gaps, out.Result.Collection.Gaps...)
		for _, ref := range out.Evidence {
			if err = st.VerifyEvidence(ctx, ref.Artifact); evaldomain.IsCode(err, "eval_evidence_unavailable") {
				out.Gaps = append(out.Gaps, "Evidence is unavailable: "+ref.ID)
			} else if err != nil {
				return err
			}
		}
		plan, err := st.FrozenPlan(ctx, owner, id)
		if err != nil {
			return err
		}
		var setup preparedSetup
		if err = json.Unmarshal(plan.Setup, &setup); err != nil {
			return err
		}
		out.Policy = setup.Checks
		for _, check := range setup.Checks {
			if check.Evaluator != "human-review@1" {
				continue
			}
			rubric, err := reviewRubric(ctx, st, e, check)
			if notFound(err) {
				out.Gaps = append(out.Gaps, "Pinned owner rubric is unavailable: "+check.ID)
				continue
			}
			if err != nil {
				return err
			}
			out.Checks = append(out.Checks, rubric)
		}
		out.Gaps = evaldomain.BoundedGaps(out.Gaps)
		return nil
	})
	return out, err
}
