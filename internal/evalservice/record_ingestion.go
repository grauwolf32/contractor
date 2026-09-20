package evalservice

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// PutRecord validates under the same owner and experiment locks as its durable
// receipt. A lost-response replay returns before re-observing mutable evidence.
func (s *Service) PutRecord(ctx context.Context, scope evalstore.Scope, id, member string, doc evaldomain.Frozen, mutation evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	operation, ok := map[string]string{"ResultInput": "result", "AssessmentInput": "assessment", "CheckRequest": "assessment"}[doc.Kind()]
	if !ok {
		return evalstore.Receipt{}, evaldomain.Failure("eval_invalid")
	}
	var receipt evalstore.Receipt
	err := pg.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		st := evalstore.NewTxStore(tx)
		var err error
		receipt, err = st.PutRecord(ctx, evalstore.RecordParams{
			Scope: scope, ExperimentID: id, MemberID: member, ActorID: scope.OwnerID, Operation: operation, Mutation: mutation,
			Build: func(e evalstore.Experiment) (evaldomain.Frozen, error) {
				plan, err := st.FrozenPlan(ctx, scope.OwnerID, id)
				if err != nil {
					return evaldomain.Frozen{}, err
				}
				m, err := st.Member(ctx, scope.OwnerID, id, member)
				if err != nil {
					return evaldomain.Frozen{}, err
				}
				if doc.Kind() == "ResultInput" {
					err = validateCollectedResult(ctx, tx, st, e, m, plan, doc)
					return doc, err
				}
				return buildAssessment(ctx, tx, st, e, m, plan, doc)
			},
		})
		return err
	})
	return receipt, err
}

func validateCollectedResult(ctx context.Context, db pg.DBTX, st *evalstore.Store, e evalstore.Experiment, member evalstore.Member, plan evalstore.Plan, doc evaldomain.Frozen) error {
	var result evaldomain.ResultInput
	if err := evaldomain.DecodeInto("ResultInput", doc.Bytes(), &result); err != nil {
		return err
	}
	if result.MemberID != member.MemberID || result.PlanSHA256 != plan.SHA256 {
		return evaldomain.Failure("eval_member_conflict")
	}
	row, err := st.ExecutionObservation(ctx, e.OwnerID, e.ID, member.MemberID)
	if err != nil {
		return err
	}
	execution := executionView(row)
	if execution.Ref == nil || !sameExecution(result.Execution, execution) {
		return evaldomain.Failure("eval_member_conflict")
	}
	inventory, err := st.Inventory(ctx, e.OwnerID, e.ID, member.MemberID)
	if err != nil {
		return err
	}
	usage, err := observedUsage(ctx, db, e.OwnerID, member.MemberID, execution, inventory)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(result.Usage, usage) {
		return evaldomain.Failure("eval_member_conflict")
	}
	outputs, err := collectOutputs(ctx, db, e.OwnerID, member, execution, inventory)
	if err != nil {
		return err
	}
	for role, ref := range result.Outputs {
		if expected, ok := outputs.Outputs[role]; !ok || expected != ref {
			return evaldomain.Failure("eval_member_conflict")
		}
	}
	for _, evidence := range result.Evidence {
		if err = st.AuthorizeEvidence(ctx, e.OwnerID, e.ID, member.MemberID, evidence.Artifact); err != nil {
			return err
		}
		if err = st.VerifyEvidence(ctx, evidence.Artifact); err != nil && !evaldomain.IsCode(err, "eval_evidence_unavailable") {
			return err
		}
	}
	return nil
}

// Reasons are presentation diagnostics. State, identity and timestamps are the
// authoritative facts, compared independently of producer wording/time zones.
func sameExecution(a, b evaldomain.ExecutionView) bool {
	a.Reason, b.Reason = nil, nil
	rawA, _ := json.Marshal(a)
	rawB, _ := json.Marshal(b)
	return string(rawA) == string(rawB)
}

func buildAssessment(ctx context.Context, db pg.DBTX, st *evalstore.Store, e evalstore.Experiment, member evalstore.Member, plan evalstore.Plan, doc evaldomain.Frozen) (evaldomain.Frozen, error) {
	var setup preparedSetup
	if err := json.Unmarshal(plan.Setup, &setup); err != nil {
		return evaldomain.Frozen{}, err
	}
	var assessment evaldomain.AssessmentInput
	var request evaldomain.CheckRequest
	if doc.Kind() == "CheckRequest" {
		if err := evaldomain.DecodeInto(doc.Kind(), doc.Bytes(), &request); err != nil {
			return evaldomain.Frozen{}, err
		}
		assessment = evaldomain.AssessmentInput{SchemaVersion: evaldomain.AssessmentSchemaVersion, Source: evaldomain.AssessmentSource{Kind: "native"}, ResultSHA256: request.ResultSHA256}
	} else {
		if err := evaldomain.DecodeInto(doc.Kind(), doc.Bytes(), &assessment); err != nil {
			return evaldomain.Frozen{}, err
		}
		if assessment.Source.Kind == "native" {
			return evaldomain.Frozen{}, evaldomain.Failure("eval_invalid")
		}
	}
	record, err := st.Record(ctx, e.OwnerID, e.ID, member.MemberID, evaldomain.RecordKindResult, assessment.ResultSHA256)
	if err != nil {
		return evaldomain.Frozen{}, err
	}
	var result evaldomain.ResultInput
	if err = json.Unmarshal(record.Document.Bytes(), &result); err != nil {
		return evaldomain.Frozen{}, err
	}
	if doc.Kind() == "CheckRequest" {
		checks := make([]evaldomain.Check, 0, len(request.CheckIDs))
		for _, check := range setup.Checks {
			if slices.Contains(request.CheckIDs, check.ID) {
				checks = append(checks, check)
			}
		}
		if len(checks) != len(request.CheckIDs) {
			return evaldomain.Frozen{}, evaldomain.Failure("eval_invalid")
		}
		assessment.Checks, err = runNativeChecks(ctx, db, checks, member.Recipe.Case.Outputs, result)
		if err != nil {
			return evaldomain.Frozen{}, err
		}
		prior, err := st.LatestRecord(ctx, e.OwnerID, e.ID, member.MemberID, evaldomain.RecordKindAssessment)
		if err != nil {
			return evaldomain.Frozen{}, err
		}
		if prior != nil {
			assessment.PreviousAssessmentSHA256 = &prior.SHA256
		}
	} else if err = validateAssessmentChecks(ctx, st, e, setup.Checks, result, assessment); err != nil {
		return evaldomain.Frozen{}, err
	}
	if assessment.Source.Kind == "human" {
		automatic := []evaldomain.Check{}
		for _, check := range setup.Checks {
			if check.Evaluator != "human-review@1" {
				automatic = append(automatic, check)
			}
		}
		checked, err := runNativeChecks(ctx, db, automatic, member.Recipe.Case.Outputs, result)
		if err != nil {
			return evaldomain.Frozen{}, err
		}
		assessment.Checks = append(assessment.Checks, checked...)
	}
	raw, err := json.Marshal(assessment)
	if err != nil {
		return evaldomain.Frozen{}, err
	}
	return evaldomain.Freeze("AssessmentInput", raw)
}

func validateAssessmentChecks(ctx context.Context, st *evalstore.Store, e evalstore.Experiment, checks []evaldomain.Check, result evaldomain.ResultInput, assessment evaldomain.AssessmentInput) error {
	pinned := map[string]evaldomain.Check{}
	for _, check := range checks {
		pinned[check.ID] = check
	}
	evidence := map[string]bool{}
	for _, ref := range result.Evidence {
		evidence[ref.ID] = true
	}
	for _, decision := range assessment.Checks {
		check, ok := pinned[decision.ID]
		if !ok || check.Evaluator != decision.Evaluator || check.ImplementationSHA256 != decision.ImplementationSHA256 {
			return evaldomain.Failure("eval_pin_mismatch")
		}
		for _, ref := range decision.EvidenceRefs {
			if !evidence[ref] {
				return evaldomain.Failure("eval_member_conflict")
			}
		}
		if assessment.Source.Kind == "human" {
			if check.Evaluator != "human-review@1" {
				return evaldomain.Failure("eval_invalid")
			}
			if _, err := reviewRubric(ctx, st, e, check); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Service) Select(ctx context.Context, scope evalstore.Scope, id string, input evaldomain.SelectionInput, mutation evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	var out evalstore.Receipt
	err := s.tx(ctx, func(st *evalstore.Store) error {
		var err error
		out, err = st.SelectRecords(ctx, scope, id, scope.OwnerID, input, mutation)
		return err
	})
	return out, err
}
