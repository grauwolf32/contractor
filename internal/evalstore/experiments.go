package evalstore

import (
	"context"
	"encoding/json"
	"reflect"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type CreateParams struct {
	Resources      []PlanResource
	Scope          Scope
	ID, PortableID string
	Document       evaldomain.Frozen
	Mutation       evaldomain.MutationIdentity
}

func (s *Store) Create(ctx context.Context, p CreateParams) (Receipt, error) {
	if !resourceID.MatchString(p.ID) || p.Document.Kind() != "CreateExperiment" {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	var input evaldomain.CreateExperiment
	if err := evaldomain.DecodeInto("CreateExperiment", p.Document.Bytes(), &input); err != nil {
		return Receipt{}, err
	}
	if err := evaldomain.Validate("Id", bytesOf(p.PortableID)); err != nil {
		return Receipt{}, err
	}
	return mutate(ctx, s, p.Scope, "experiments", "experiment-create", p.Mutation, func() (ExperimentReceipt, error) {
		var draft []byte
		var datasetID, datasetRevision *string
		var budgets evaldomain.Budgets
		state := evaldomain.StateDraft
		if input.Draft != nil {
			draft = bytesOf(input.Draft)
			datasetID = &input.Draft.Dataset.ID
			datasetRevision = &input.Draft.Dataset.Revision
			budgets = input.Draft.Budgets
			if _, err := s.Dataset(ctx, p.Scope, *datasetID, *datasetRevision); err != nil {
				return ExperimentReceipt{}, err
			}
		} else {
			budgets = input.Registration.Budgets
			state = "ready"
			if p.PortableID != input.Registration.Manifest.ExperimentID {
				return ExperimentReceipt{}, evaldomain.Failure("eval_member_conflict")
			}
		}
		_, err := s.db.Exec(ctx, `
INSERT INTO eval_experiments(experiment_id, owner_id, project_id, portable_id, control_mode, name, state, draft, dataset_id, dataset_revision, max_in_flight, wall_ms, token_limit)
VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
`, p.ID, p.Scope.OwnerID, p.Scope.ProjectID, p.PortableID, input.ControlMode, input.Name, state, draft, datasetID, datasetRevision, budgets.MaxInFlight, budgets.WallMS, budgets.MaxObservedTotalTokens)
		if err != nil {
			return ExperimentReceipt{}, normalize(err)
		}
		if _, err = s.db.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) VALUES($1)`, p.ID); err != nil {
			return ExperimentReceipt{}, err
		}
		revision := int64(1)
		if input.Registration != nil {
			reg, err := evaldomain.Freeze("ExternalRegistration", bytesOf(input.Registration))
			if err != nil {
				return ExperimentReceipt{}, err
			}
			setup := bytesOf(map[string]any{
				"variants":   input.Registration.Variants,
				"checks":     input.Registration.Checks,
				"comparison": input.Registration.Comparison,
				"budgets":    budgets,
				"source":     input.Registration.Source,
			})
			recipes := make(map[string]evaldomain.Case, len(input.Registration.Recipes))
			for _, r := range input.Registration.Recipes {
				recipes[r.MemberID] = r.Case
			}
			e, err := s.locked(ctx, p.Scope, p.ID)
			if err != nil {
				return ExperimentReceipt{}, err
			}
			if err = s.persistPlan(ctx, e, reg, setup, recipes); err != nil {
				return ExperimentReceipt{}, err
			}
			if err = s.putResources(ctx, p.ID, p.Resources); err != nil {
				return ExperimentReceipt{}, err
			}
			revision++
		}
		return ExperimentReceipt{ExperimentID: p.ID, Revision: revision, State: state}, nil
	})
}

func (s *Store) UpdateDraft(ctx context.Context, scope Scope, id string, document evaldomain.Frozen, mutation evaldomain.MutationIdentity) (Receipt, error) {
	if document.Kind() != "DraftUpdate" {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	var input evaldomain.DraftUpdate
	if err := evaldomain.DecodeInto("DraftUpdate", document.Bytes(), &input); err != nil {
		return Receipt{}, err
	}
	return mutate(ctx, s, scope, id, "draft-update", mutation, func() (ExperimentReceipt, error) {
		e, err := s.locked(ctx, scope, id)
		if err != nil {
			return ExperimentReceipt{}, err
		}
		if err = checkMutable(e, mutation); err != nil {
			return ExperimentReceipt{}, err
		}
		if e.ControlMode != evaldomain.ControlServer {
			return ExperimentReceipt{}, evaldomain.Failure("eval_external_control")
		}
		if e.State != evaldomain.StateDraft {
			return ExperimentReceipt{}, evaldomain.Failure("eval_not_ready")
		}
		if _, err = s.Dataset(ctx, scope, input.Draft.Dataset.ID, input.Draft.Dataset.Revision); err != nil {
			return ExperimentReceipt{}, err
		}
		_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET name=$2,draft=$3,dataset_id=$4,dataset_revision=$5,max_in_flight=$6,wall_ms=$7,token_limit=$8,`+advance+` WHERE experiment_id=$1`, id, input.Name, bytesOf(input.Draft), input.Draft.Dataset.ID, input.Draft.Dataset.Revision, input.Draft.Budgets.MaxInFlight, input.Draft.Budgets.WallMS, input.Draft.Budgets.MaxObservedTotalTokens)
		return ExperimentReceipt{ExperimentID: id, Revision: e.Revision + 1, State: e.State}, normalize(err)
	})
}

type Recipe struct {
	Case    evaldomain.ExecutionCase `json:"case"`
	Variant evaldomain.Variant       `json:"variant"`
}

type Member struct {
	evaldomain.PublicMember
	PairID                       string
	Ordinal                      int
	ExecutionKind, SubmissionKey string
	Recipe                       Recipe
}

type Plan struct {
	SHA256 string

	Document evaldomain.Frozen `json:"-"`
	Setup    json.RawMessage   `json:"-"`
}

// FrozenPlan trusts the stored document: persistPlan writes only Freeze output,
// eval_plans_immutable rejects every later change and document_sha256 is
// generated from the stored bytes, so ticks never revalidate up to 1 MiB.
func (s *Store) FrozenPlan(ctx context.Context, owner, id string) (Plan, error) {
	var kind, digest, identity string
	var raw, setup []byte
	err := s.db.QueryRow(ctx, `
SELECT p.document_kind, p.document, p.document_sha256, p.setup, p.plan_sha256
FROM eval_frozen_plans p
JOIN eval_experiments e USING(experiment_id)
WHERE e.owner_id=$1
    AND e.experiment_id=$2
`, owner, id).Scan(&kind, &raw, &digest, &setup, &identity)
	if err != nil {
		return Plan{}, normalize(err)
	}
	return Plan{Document: evaldomain.Restore(kind, raw, digest), Setup: setup, SHA256: identity}, nil
}

// PlanMetadata contains only the immutable attribution and setup used by
// admission and per-member collection. It never transfers the matrix document.
type PlanMetadata struct {
	SHA256 string
	Setup  json.RawMessage `json:"-"`
}

func (s *Store) FrozenPlanMetadata(ctx context.Context, owner, id string) (PlanMetadata, error) {
	var out PlanMetadata
	err := s.db.QueryRow(ctx, `
SELECT p.plan_sha256, p.setup
FROM eval_frozen_plans p JOIN eval_experiments e USING (experiment_id)
WHERE e.owner_id = $1 AND e.experiment_id = $2`, owner, id).Scan(&out.SHA256, &out.Setup)
	return out, normalize(err)
}

// FreezePrepared is called after read-only preparation, under the coordinator's
// claim. The whole expected matrix and its exact bytes commit together.
func (s *Store) FreezePrepared(ctx context.Context, scope Scope, id string, claim Claim, document evaldomain.Frozen, setup []byte, cases map[string]evaldomain.Case, resources ...PlanResource) error {
	if err := s.requireTx(); err != nil {
		return err
	}
	active, err := s.project(ctx, scope, true)
	if err != nil {
		return err
	}
	if !active {
		return evaldomain.Failure("eval_project_deleting")
	}
	e, err := s.locked(ctx, scope, id)
	if err != nil {
		return err
	}
	if err = s.checkClaim(ctx, id, claim); err != nil {
		return err
	}
	if e.DeletionRequestedAt != nil {
		return evaldomain.Failure("eval_project_deleting")
	}
	if e.ControlMode != evaldomain.ControlServer || e.State != evaldomain.StatePreparing || document.Kind() != "playground.plan/v1" {
		return evaldomain.Failure("eval_not_ready")
	}
	if err = s.persistPlan(ctx, e, document, setup, cases); err != nil {
		return err
	}
	return s.putResources(ctx, id, resources)
}

func (s *Store) persistPlan(ctx context.Context, e Experiment, document evaldomain.Frozen, setup []byte, cases map[string]evaldomain.Case) error {
	if err := evaldomain.Validate("ExperimentSetup", setup); err != nil {
		return err
	}
	var settings struct {
		Variants []evaldomain.Variant `json:"variants"`
		Budgets  evaldomain.Budgets   `json:"budgets"`
	}
	if err := json.Unmarshal(setup, &settings); err != nil {
		return err
	}
	if settings.Budgets.MaxInFlight != e.MaxInFlight || settings.Budgets.WallMS != e.WallMS || !reflect.DeepEqual(settings.Budgets.MaxObservedTotalTokens, e.TokenLimit) {
		return evaldomain.Failure("eval_pin_mismatch")
	}
	identity := document.Digest()
	var manifest evaldomain.PublicPlan
	var order []string
	reasons := map[string]*string{}
	if document.Kind() == "ExternalRegistration" {
		var reg evaldomain.ExternalRegistration
		if err := json.Unmarshal(document.Bytes(), &reg); err != nil {
			return err
		}
		manifest = reg.Manifest
		identity = reg.SourcePlanSHA256
	} else {
		var err error
		manifest, err = evaldomain.PublicPlanProjection(document)
		if err != nil {
			return err
		}
		var native struct {
			Order   []string `json:"execution_order"`
			Members []struct {
				ID     string  `json:"member_id"`
				Reason *string `json:"reason"`
			} `json:"members"`
		}
		if err = json.Unmarshal(document.Bytes(), &native); err != nil {
			return err
		}
		order = native.Order
		for _, member := range native.Members {
			reasons[member.ID] = member.Reason
		}
	}
	if manifest.ExperimentID != e.PortableID || len(manifest.Members) != len(cases) || len(manifest.Members) > settings.Budgets.MaxMembers {
		return evaldomain.Failure("eval_member_conflict")
	}
	variants := map[string]evaldomain.Variant{}
	for _, v := range settings.Variants {
		variants[v.ID] = v
	}
	if len(variants) != 2 || settings.Variants[0].Kind != settings.Variants[1].Kind {
		return evaldomain.Failure("eval_invalid")
	}
	ordinals := map[string]int{}
	if order == nil {
		for _, m := range manifest.Members {
			order = append(order, m.MemberID)
		}
	}
	if len(order) != len(manifest.Members) {
		return evaldomain.Failure("eval_member_conflict")
	}
	for i, id := range order {
		if _, ok := ordinals[id]; ok {
			return evaldomain.Failure("eval_member_conflict")
		}
		ordinals[id] = i
	}
	change := observe
	if e.ControlMode == evaldomain.ControlExternal {
		// External registration authors the plan within Create, and its accepted
		// revision has always named the second authority write.
		change = advance
	}
	_, err := s.db.Exec(ctx, `UPDATE eval_experiments SET state='ready',diagnostic=NULL,expected_count=$2,`+change+` WHERE experiment_id=$1`, e.ID, len(manifest.Members))
	if err != nil {
		return err
	}
	_, err = s.db.Exec(ctx, `INSERT INTO eval_frozen_plans(experiment_id,document_kind,document,setup,plan_sha256) VALUES($1,$2,$3,$4,$5)`, e.ID, document.Kind(), document.Bytes(), setup, identity)
	if err != nil {
		return normalize(err)
	}
	for _, m := range manifest.Members {
		c, ok := cases[m.MemberID]
		if !ok || c.ID != m.CaseID {
			return evaldomain.Failure("eval_member_conflict")
		}
		v, ok := variants[m.VariantID]
		if !ok {
			return evaldomain.Failure("eval_member_conflict")
		}
		ordinal, ok := ordinals[m.MemberID]
		if !ok {
			return evaldomain.Failure("eval_member_conflict")
		}
		visible, err := evaldomain.ExecutionProjection(c)
		if err != nil {
			return err
		}
		kind := "run"
		if v.Kind == "audit" {
			kind = "audit"
		}
		pair, err := evaldomain.PairID(e.PortableID, m.SuiteID, m.CaseID, m.Sample)
		if err != nil {
			return err
		}
		// Scope the effect key by the server identity, not the portable ID alone.
		key := "eval-" + evaldomain.Digest(bytesOf([]string{e.ID, m.MemberID}))[7:]
		_, err = s.db.Exec(ctx, `
INSERT INTO eval_members(experiment_id, member_id, pair_id, ordinal, suite_id, case_id, sample, variant_id, eligibility, execution_kind, recipe, submission_key, case_sha256, binding_sha256, eligibility_reason)
VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
`, e.ID, m.MemberID, pair, ordinal, m.SuiteID, m.CaseID, m.Sample, m.VariantID, m.Eligibility, kind, bytesOf(Recipe{visible, v}), key, m.CaseSHA256, m.BindingSHA256, reasons[m.MemberID])
		if err != nil {
			return normalize(err)
		}
	}
	return nil
}
