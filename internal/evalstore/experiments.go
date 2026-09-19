package evalstore

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"reflect"
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
	return s.mutate(ctx, p.Scope, "experiments", "experiment-create", p.Mutation, func() (Reference, error) {
		var draft []byte
		var datasetID, datasetRevision *string
		var budgets evaldomain.Budgets
		state := "draft"
		if input.Draft != nil {
			draft = bytesOf(input.Draft)
			datasetID = &input.Draft.Dataset.ID
			datasetRevision = &input.Draft.Dataset.Revision
			budgets = input.Draft.Budgets
			if _, err := s.Dataset(ctx, p.Scope, *datasetID, *datasetRevision); err != nil {
				return Reference{}, err
			}
		} else {
			budgets = input.Registration.Budgets
			state = "ready"
			if p.PortableID != input.Registration.Manifest.ExperimentID {
				return Reference{}, evaldomain.Failure("eval_member_conflict")
			}
		}
		_, err := s.db.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,draft,dataset_id,dataset_revision,max_in_flight,wall_ms,token_limit) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)`, p.ID, p.Scope.OwnerID, p.Scope.ProjectID, p.PortableID, input.ControlMode, input.Name, state, draft, datasetID, datasetRevision, budgets.MaxInFlight, budgets.WallMS, budgets.MaxObservedTotalTokens)
		if err != nil {
			return Reference{}, normalize(err)
		}
		if _, err = s.db.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) VALUES($1)`, p.ID); err != nil {
			return Reference{}, err
		}
		revision := int64(1)
		if input.Registration != nil {
			reg, err := evaldomain.Freeze("ExternalRegistration", bytesOf(input.Registration))
			if err != nil {
				return Reference{}, err
			}
			setup := bytesOf(map[string]any{"variants": input.Registration.Variants, "checks": input.Registration.Checks, "comparison": input.Registration.Comparison, "budgets": budgets, "source": input.Registration.Source})
			recipes := make(map[string]evaldomain.Case, len(input.Registration.Recipes))
			for _, r := range input.Registration.Recipes {
				recipes[r.MemberID] = r.Case
			}
			e, err := s.locked(ctx, p.Scope, p.ID)
			if err != nil {
				return Reference{}, err
			}
			if err = s.persistPlan(ctx, e, reg, setup, recipes); err != nil {
				return Reference{}, err
			}
			if err = s.putResources(ctx, p.ID, p.Resources); err != nil {
				return Reference{}, err
			}
			revision++
		}
		return Reference{ID: p.ID, Revision: revision, State: state}, nil
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
	return s.mutate(ctx, scope, id, "draft-update", mutation, func() (Reference, error) {
		e, err := s.locked(ctx, scope, id)
		if err != nil {
			return Reference{}, err
		}
		if err = checkMutable(e, mutation); err != nil {
			return Reference{}, err
		}
		if e.ControlMode != "server" {
			return Reference{}, evaldomain.Failure("eval_external_control")
		}
		if e.State != "draft" {
			return Reference{}, evaldomain.Failure("eval_not_ready")
		}
		if _, err = s.Dataset(ctx, scope, input.Draft.Dataset.ID, input.Draft.Dataset.Revision); err != nil {
			return Reference{}, err
		}
		_, err = s.db.Exec(ctx, `UPDATE eval_experiments SET name=$2,draft=$3,dataset_id=$4,dataset_revision=$5,max_in_flight=$6,wall_ms=$7,token_limit=$8,`+advance+` WHERE experiment_id=$1`, id, input.Name, bytesOf(input.Draft), input.Draft.Dataset.ID, input.Draft.Dataset.Revision, input.Draft.Budgets.MaxInFlight, input.Draft.Budgets.WallMS, input.Draft.Budgets.MaxObservedTotalTokens)
		return Reference{ID: id, Revision: e.Revision + 1, State: e.State}, normalize(err)
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

func (s *Store) FrozenPlan(ctx context.Context, owner, id string) (Plan, error) {
	var kind, identity string
	var raw, setup []byte
	err := s.db.QueryRow(ctx, `SELECT p.document_kind,p.document,p.setup,p.plan_sha256 FROM eval_frozen_plans p JOIN eval_experiments e USING(experiment_id) WHERE e.owner_id=$1 AND e.experiment_id=$2`, owner, id).Scan(&kind, &raw, &setup, &identity)
	if err != nil {
		return Plan{}, normalize(err)
	}
	doc, err := evaldomain.Freeze(kind, raw)
	return Plan{Document: doc, Setup: setup, SHA256: identity}, err
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
	if e.ControlMode != "server" || e.State != "preparing" || document.Kind() != "playground.plan/v1" {
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
			Order []string `json:"execution_order"`
		}
		if err = json.Unmarshal(document.Bytes(), &native); err != nil {
			return err
		}
		order = native.Order
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
	_, err := s.db.Exec(ctx, `UPDATE eval_experiments SET state='ready',expected_count=$2,`+advance+` WHERE experiment_id=$1`, e.ID, len(manifest.Members))
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
		_, err = s.db.Exec(ctx, `INSERT INTO eval_members(experiment_id,member_id,pair_id,ordinal,suite_id,case_id,sample,variant_id,eligibility,execution_kind,recipe,submission_key,case_sha256,binding_sha256) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)`, e.ID, m.MemberID, pair, ordinal, m.SuiteID, m.CaseID, m.Sample, m.VariantID, m.Eligibility, kind, bytesOf(Recipe{visible, v}), key, m.CaseSHA256, m.BindingSHA256)
		if err != nil {
			return normalize(err)
		}
	}
	return nil
}

// Member pages use frozen order and a hard SQL LIMIT before recipe decoding.
func (s *Store) Members(ctx context.Context, owner, id string, afterOrdinal, limit int) ([]Member, error) {
	if limit < 1 || limit > evaldomain.MaxPageSize || afterOrdinal < -1 {
		return nil, evaldomain.Failure("eval_invalid")
	}
	if _, err := s.Get(ctx, owner, id); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `SELECT m.member_id,m.pair_id,m.ordinal,m.suite_id,m.case_id,m.sample,m.variant_id,m.eligibility,m.execution_kind,m.recipe,m.submission_key,m.case_sha256,m.binding_sha256 FROM eval_members m WHERE m.experiment_id=$1 AND m.ordinal>$2 ORDER BY m.ordinal LIMIT $3`, id, afterOrdinal, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Member, 0)
	for rows.Next() {
		var m Member
		var recipe []byte
		if err = rows.Scan(&m.MemberID, &m.PairID, &m.Ordinal, &m.SuiteID, &m.CaseID, &m.Sample, &m.VariantID, &m.Eligibility, &m.ExecutionKind, &recipe, &m.SubmissionKey, &m.CaseSHA256, &m.BindingSHA256); err != nil {
			return nil, err
		}
		if err = json.Unmarshal(recipe, &m.Recipe); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}
