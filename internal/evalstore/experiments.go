package evalstore

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type CreateParams struct {
	Resources      []PlanResource
	Scope          Scope
	ID, PortableID string
	Document       evaldomain.Frozen
	Mutation       evaldomain.MutationIdentity
}

func (s *Store) Create(ctx context.Context, p CreateParams) (Receipt, error) {
	if !contracts.ValidResourceID(p.ID) || p.Document.Kind() != "CreateExperiment" {
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

// Recipe is a member's execution recipe. persistPlan stores it as the
// json.Marshal encoding of this struct, assembled from memberRows' case and
// variant parts; a change to its fields or tags changes that split too.
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
// generated from the stored bytes when the document is written, so ticks never
// revalidate or rehash up to 1 MiB.
func (s *Store) FrozenPlan(ctx context.Context, owner, id string) (Plan, error) {
	var kind, digest, identity string
	var raw, setup []byte
	err := s.db.QueryRow(ctx, `
SELECT p.plan_kind, p.plan_document, p.plan_document_sha256, p.plan_setup, p.plan_sha256
FROM eval_experiments p
WHERE p.plan_document IS NOT NULL AND p.owner_id=$1
    AND p.experiment_id=$2
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
SELECT p.plan_sha256, p.plan_setup
FROM eval_experiments p
WHERE p.plan_document IS NOT NULL AND p.owner_id = $1 AND p.experiment_id = $2`, owner, id).Scan(&out.SHA256, &out.Setup)
	return out, normalize(err)
}

// PlanView is what the experiment view shows of its frozen plan: the plan
// metadata and, for a native plan, the readiness its freeze stored. An
// external registration has none.
type PlanView struct {
	PlanMetadata
	Readiness *evaldomain.Readiness
}

// FrozenPlanView reads the plan metadata and stored readiness without the
// matrix document (EVM-R242). The readiness must decode strictly, with every
// member present and none unknown; a malformed stored copy fails the read as
// an internal error. The caller still validates the response that carries it.
func (s *Store) FrozenPlanView(ctx context.Context, owner, id string) (PlanView, error) {
	var out PlanView
	var readiness []byte
	err := s.db.QueryRow(ctx, `
SELECT p.plan_sha256, p.plan_setup, p.plan_readiness
FROM eval_experiments p
WHERE p.plan_document IS NOT NULL AND p.owner_id = $1 AND p.experiment_id = $2`, owner, id).Scan(&out.SHA256, &out.Setup, &readiness)
	if err != nil {
		return PlanView{}, normalize(err)
	}
	if readiness != nil {
		out.Readiness = &evaldomain.Readiness{}
		if err = strictjson.Decode(readiness, out.Readiness); err != nil {
			return PlanView{}, fmt.Errorf("stored readiness of eval experiment %s does not decode: %w", id, err)
		}
	}
	return out, nil
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
		Variants   []evaldomain.Variant  `json:"variants"`
		Budgets    evaldomain.Budgets    `json:"budgets"`
		Comparison evaldomain.Comparison `json:"comparison"`
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
	var readiness []byte
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
		// Readiness is immutable with the plan, so the view stores it once
		// instead of reading the document to recompute it (EVM-R242).
		view, err := evaldomain.NativeReadiness(document, settings.Comparison)
		if err != nil {
			return err
		}
		readiness = bytesOf(view)
		if err = evaldomain.Validate("Readiness", readiness); err != nil {
			return err
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
	rows, err := prepareMembers(e, manifest.Members, cases, settings.Variants, ordinals, reasons)
	if err != nil {
		return err
	}
	change := observe
	if e.ControlMode == evaldomain.ControlExternal {
		// External registration authors the plan within Create, and its accepted
		// revision has always named the second authority write.
		change = advance
	}
	_, err = s.db.Exec(ctx, `
UPDATE eval_experiments
SET state='ready', diagnostic=NULL, expected_count=$2,
    plan_kind=$3, plan_document=$4, plan_setup=$5, plan_sha256=$6,
    plan_readiness=$7, plan_created_at=clock_timestamp(), `+change+`
WHERE experiment_id=$1`, e.ID, len(manifest.Members), document.Kind(), document.Bytes(), setup, identity, readiness)
	if err != nil {
		return normalize(err)
	}
	start := 0
	for _, end := range freezeBatches(len(rows.ids), rows.recipeBytes) {
		if _, err = s.db.Exec(ctx, insertMembersSQL, rows.arguments(e.ID, start, end)...); err != nil {
			return normalize(err)
		}
		start = end
	}
	return nil
}

// Each freeze insert statement holds at most freezeBatchRows rows and
// freezeBatchBytes stored recipe or resource document bytes, but at least one
// row (LIM-EVM-18). A member statement's trigger keeps every inserted row,
// recipe uncompressed, in its transition table, so the byte bound keeps that
// table within PostgreSQL's default 4 MB work_mem and each statement far below
// the 15 s default statement timeout even for the largest admitted recipes.
// The row bound equals the native member ceiling (LIM-EVM-3), so a native plan
// of ordinary recipes still freezes with one statement.
const (
	freezeBatchRows  = 1000
	freezeBatchBytes = 4 << 20
)

// freezeBatches splits n rows, in order, into consecutive insert batches and
// returns the end index of each. size(i) is the stored size of row i; a row
// that alone exceeds freezeBatchBytes forms its own batch.
func freezeBatches(n int, size func(int) int) []int {
	var ends []int
	rows, bytes := 0, 0
	for i := range n {
		next := size(i)
		if rows > 0 && (rows == freezeBatchRows || bytes+next > freezeBatchBytes) {
			ends = append(ends, i)
			rows, bytes = 0, 0
		}
		rows++
		bytes += next
	}
	if rows > 0 {
		ends = append(ends, n)
	}
	return ends
}

// memberRows is the expected matrix as one array per eval_members column,
// validated before any write. It has one row per manifest member, at most
// the setup's maxMembers budget (LIM-EVF-8). A stored recipe is
// bytesOf(Recipe{case, variant}), whose encoding is the case part
// {"case":<case>,"variant": followed by the variant part <variant>}. Each
// distinct projected case and each variant is kept once, and each insert
// batch sends only the parts its members use.
type memberRows struct {
	ids, pairs, suites, caseIDs, variantIDs  []string
	eligibility, kinds, keys                 []string
	caseDigests, bindingDigests              []string
	ordinals, samples, caseRefs, variantRefs []int
	reasons                                  []*string
	caseParts, variantParts                  [][]byte
}

func prepareMembers(e Experiment, members []evaldomain.PublicMember, cases map[string]evaldomain.Case, variants []evaldomain.Variant, ordinals map[string]int, reasons map[string]*string) (memberRows, error) {
	n := len(members)
	rows := memberRows{
		ids: make([]string, 0, n), pairs: make([]string, 0, n), suites: make([]string, 0, n),
		caseIDs: make([]string, 0, n), variantIDs: make([]string, 0, n), eligibility: make([]string, 0, n),
		kinds: make([]string, 0, n), keys: make([]string, 0, n), caseDigests: make([]string, 0, n),
		bindingDigests: make([]string, 0, n), ordinals: make([]int, 0, n), samples: make([]int, 0, n),
		caseRefs: make([]int, 0, n), variantRefs: make([]int, 0, n), reasons: make([]*string, 0, n),
	}
	variantRef := make(map[string]int, len(variants))
	kinds := make(map[string]string, len(variants))
	for _, v := range variants {
		rows.variantParts = append(rows.variantParts, append(bytesOf(v), '}'))
		variantRef[v.ID] = len(rows.variantParts)
		kinds[v.ID] = "run"
		if v.Kind == "audit" {
			kinds[v.ID] = "audit"
		}
	}
	caseRef := map[string]int{}
	for _, m := range members {
		c, ok := cases[m.MemberID]
		if !ok || c.ID != m.CaseID {
			return memberRows{}, evaldomain.Failure("eval_member_conflict")
		}
		variant, ok := variantRef[m.VariantID]
		if !ok {
			return memberRows{}, evaldomain.Failure("eval_member_conflict")
		}
		ordinal, ok := ordinals[m.MemberID]
		if !ok {
			return memberRows{}, evaldomain.Failure("eval_member_conflict")
		}
		// ExecutionProjection depends only on the case's JSON encoding, so
		// members that share a case share one validated case part.
		encoded := string(bytesOf(c))
		part, ok := caseRef[encoded]
		if !ok {
			visible, err := evaldomain.ExecutionProjection(c)
			if err != nil {
				return memberRows{}, err
			}
			rows.caseParts = append(rows.caseParts, slices.Concat([]byte(`{"case":`), bytesOf(visible), []byte(`,"variant":`)))
			part = len(rows.caseParts)
			caseRef[encoded] = part
		}
		pair, err := evaldomain.PairID(e.PortableID, m.SuiteID, m.CaseID, m.Sample)
		if err != nil {
			return memberRows{}, err
		}
		// Scope the effect key by the server identity, not the portable ID alone.
		key := "eval-" + evaldomain.Digest(bytesOf([]string{e.ID, m.MemberID}))[7:]
		rows.ids = append(rows.ids, m.MemberID)
		rows.pairs = append(rows.pairs, pair)
		rows.ordinals = append(rows.ordinals, ordinal)
		rows.suites = append(rows.suites, m.SuiteID)
		rows.caseIDs = append(rows.caseIDs, m.CaseID)
		rows.samples = append(rows.samples, m.Sample)
		rows.variantIDs = append(rows.variantIDs, m.VariantID)
		rows.eligibility = append(rows.eligibility, m.Eligibility)
		rows.kinds = append(rows.kinds, kinds[m.VariantID])
		rows.keys = append(rows.keys, key)
		rows.caseDigests = append(rows.caseDigests, m.CaseSHA256)
		rows.bindingDigests = append(rows.bindingDigests, m.BindingSHA256)
		rows.reasons = append(rows.reasons, reasons[m.MemberID])
		rows.caseRefs = append(rows.caseRefs, part)
		rows.variantRefs = append(rows.variantRefs, variant)
	}
	return rows, nil
}

// recipeBytes is the stored recipe size of member i.
func (r memberRows) recipeBytes(i int) int {
	return len(r.caseParts[r.caseRefs[i]-1]) + len(r.variantParts[r.variantRefs[i]-1])
}

// arguments returns insertMembersSQL's parameters for members [start, end)
// in placeholder order. The batch carries only the case and variant parts its
// members use, renumbered from 1.
func (r memberRows) arguments(experimentID string, start, end int) []any {
	caseRefs, variantRefs := make([]int, 0, end-start), make([]int, 0, end-start)
	var caseParts, variantParts [][]byte
	cases, variants := map[int]int{}, map[int]int{}
	for i := start; i < end; i++ {
		part, ok := cases[r.caseRefs[i]]
		if !ok {
			caseParts = append(caseParts, r.caseParts[r.caseRefs[i]-1])
			part = len(caseParts)
			cases[r.caseRefs[i]] = part
		}
		caseRefs = append(caseRefs, part)
		part, ok = variants[r.variantRefs[i]]
		if !ok {
			variantParts = append(variantParts, r.variantParts[r.variantRefs[i]-1])
			part = len(variantParts)
			variants[r.variantRefs[i]] = part
		}
		variantRefs = append(variantRefs, part)
	}
	return []any{
		experimentID, r.ids[start:end], r.pairs[start:end], r.ordinals[start:end],
		r.suites[start:end], r.caseIDs[start:end], r.samples[start:end],
		r.variantIDs[start:end], r.eligibility[start:end], r.kinds[start:end],
		r.keys[start:end], r.caseDigests[start:end], r.bindingDigests[start:end],
		r.reasons[start:end], caseRefs, variantRefs, caseParts, variantParts,
	}
}
