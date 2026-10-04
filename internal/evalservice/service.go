package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Each tick rotates a bounded set of members so large experiments share the worker.
const membersPerTick = 10

type BindingResolver interface {
	Resolve(context.Context, string, evaldomain.Variant, []evaldomain.Case) (Preflight, error)
}

// ExecutionDriver advances only a previously admitted member. Choosing another
// member is the native coordinator's separate responsibility.
type ExecutionDriver interface {
	Reconcile(context.Context, evalstore.Experiment, evalstore.Member, evalstore.Claim) error
}

type Options struct {
	Pool     *pgxpool.Pool
	Resolver BindingResolver
	Driver   ExecutionDriver
	Now      func() time.Time
}

type Service struct {
	pool     *pgxpool.Pool
	resolver BindingResolver
	driver   ExecutionDriver
	now      func() time.Time
}

func New(o Options) (*Service, error) {
	if o.Pool == nil || o.Resolver == nil || o.Driver == nil {
		return nil, errors.New("eval service dependencies are incomplete")
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	return &Service{o.Pool, o.Resolver, o.Driver, o.Now}, nil
}

func (s *Service) tx(ctx context.Context, fn func(*evalstore.Store) error) error {
	return pg.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error { return fn(evalstore.NewTxStore(tx)) })
}

func scope(e evalstore.Experiment) evalstore.Scope {
	return evalstore.Scope{OwnerID: e.OwnerID, ProjectID: e.ProjectID}
}

func diagnostic(code string) json.RawMessage {
	d := evaldomain.Failure(code)
	b, _ := json.Marshal(safeDiagnostic{Code: d.Code, Recovery: d.Recovery})
	return b
}

func (s *Service) prepare(ctx context.Context, e evalstore.Experiment, claim evalstore.Claim) error {
	var draft evaldomain.Draft
	if err := evaldomain.DecodeInto("Draft", e.Draft.Bytes(), &draft); err != nil {
		return err
	}
	dataset, err := evalstore.NewPostgresStore(s.pool).Dataset(ctx, scope(e), draft.Dataset.ID, draft.Dataset.Revision)
	if err != nil {
		return err
	}
	var data evaldomain.DatasetInput
	if err = evaldomain.DecodeInto("DatasetInput", dataset.Document.Bytes(), &data); err != nil {
		return err
	}
	wanted := map[string]bool{}
	for _, id := range draft.CaseIDs {
		wanted[id] = true
	}
	cases := []evaldomain.Case{}
	for _, c := range data.Cases {
		if wanted[c.ID] {
			cases = append(cases, c)
		}
	}
	if len(cases) != len(wanted) {
		return evaldomain.Failure("eval_not_found")
	}
	resolved := map[string]Preflight{}
	for _, v := range draft.Variants {
		r, err := s.resolver.Resolve(ctx, e.OwnerID, v, cases)
		if err != nil {
			return err
		}
		resolved[v.ID] = r
	}
	bundle, err := BuildPlan(e.PortableID, s.now(), draft, data, resolved)
	if err != nil {
		return err
	}
	resources := make([]evalstore.PlanResource, 0, len(bundle.Resources)+len(bundle.Private)+len(bundle.Inputs))
	for _, r := range bundle.Resources {
		resources = append(resources, evalstore.PlanResource{Path: r.Path, Document: r.Document})
	}
	for path, doc := range bundle.Private {
		resources = append(resources, evalstore.PlanResource{Path: path, Document: doc})
	}
	for path, ref := range bundle.Inputs {
		b, err := jsonBytes(ref)
		if err != nil {
			return err
		}
		doc, err := evaldomain.Freeze("Artifact", b)
		if err != nil {
			return err
		}
		resources = append(resources, evalstore.PlanResource{Path: path, Document: doc})
	}
	return s.tx(ctx, func(store *evalstore.Store) error {
		return store.FreezePrepared(ctx, scope(e), e.ID, claim, bundle.Plan, bundle.Setup, bundle.Cases, resources...)
	})
}

// Tick performs a bounded amount of reconciliation. External mode never enters
// the member selection/admission path: it only recovers already accepted work.
// A member whose reconciliation fails is reported only after admission,
// lifecycle progress and view publication ran, so it cannot stall the rest.
func (s *Service) Tick(ctx context.Context, claim evalstore.Claim) (bool, error) {
	progressed, memberErr, err := s.tickExecution(ctx, claim)
	if err == nil {
		err = s.publish(ctx, claim)
	}
	return progressed, errors.Join(memberErr, err)
}

// publish collects the claimed experiment's view unless the tick purged it.
func (s *Service) publish(ctx context.Context, claim evalstore.Claim) error {
	store := evalstore.NewPostgresStore(s.pool)
	e, err := store.GetClaimed(ctx, claim)
	if errors.Is(err, evalstore.ErrClaimLost) {
		exists, checkErr := store.ClaimTargetExists(ctx, claim)
		if checkErr != nil || !exists {
			return checkErr
		}
	}
	if notFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	return s.CollectView(ctx, e, claim)
}

// tickExecution returns the joined reconcile failures of individual members
// separately from a failure that ends the tick.
func (s *Service) tickExecution(ctx context.Context, claim evalstore.Claim) (progressed bool, memberErr, err error) {
	store := evalstore.NewPostgresStore(s.pool)
	e, err := store.GetClaimed(ctx, claim)
	if err != nil {
		return false, nil, err
	}
	if err = s.recoverCommands(ctx, e, claim); err != nil {
		return false, nil, err
	}
	if e.State == evaldomain.StatePreparing {
		if err = s.prepare(ctx, e, claim); err != nil {
			progressed, err = s.preparationFailed(ctx, e, claim, err)
			return progressed, nil, err
		}
		return true, nil, s.finishCommands(ctx, e, claim, true, nil, evaldomain.CommandPrepare)
	}
	outstanding, err := store.ReconciliationCandidates(ctx, e.OwnerID, e.ID, membersPerTick)
	if err != nil {
		return false, nil, err
	}
	var memberErrors []error
	for _, memberID := range outstanding {
		m, err := store.Member(ctx, e.OwnerID, e.ID, memberID)
		if err != nil {
			return false, errors.Join(memberErrors...), err
		}
		if err = s.driver.Reconcile(ctx, e, m, claim); err != nil {
			memberErrors = append(memberErrors, err)
			// Rotate failed observations as well: one uncertain member must not
			// starve cancellation/recovery of the rest of a large experiment.
			if ctx.Err() == nil {
				if rotateErr := s.tx(ctx, func(st *evalstore.Store) error { return st.ObserveTokens(ctx, scope(e), e.ID, m.MemberID, claim, 0) }); rotateErr != nil {
					return false, errors.Join(memberErrors...), rotateErr
				}
			}
		}
	}
	memberErr = errors.Join(memberErrors...)
	if e, err = store.Get(ctx, e.OwnerID, e.ID); err != nil {
		return false, memberErr, err
	}
	progressed, err = s.advance(ctx, e, claim, len(outstanding) > 0)
	return progressed, memberErr, err
}

// advance applies budget stops, native admission and drain transitions to the
// state that member reconciliation left. A failed member stays outstanding, so
// it still holds its capacity and blocks only the drain that must wait for it.
func (s *Service) advance(ctx context.Context, e evalstore.Experiment, claim evalstore.Claim, reconciled bool) (bool, error) {
	var err error
	expired := e.DeadlineAt != nil && !s.now().Before(*e.DeadlineAt)
	exhausted := e.TokenLimit != nil && e.ObservedTokens >= *e.TokenLimit
	if target, stop := e.Lifecycle().BudgetStop(expired || exhausted); stop {
		return true, s.tx(ctx, func(st *evalstore.Store) error {
			return st.Transition(ctx, scope(e), e.ID, claim, e.State, target, e.ObservedTokens, diagnostic("eval_budget_exhausted"))
		})
	}
	switch e.State {
	case evaldomain.StateRunning:
		if err = s.finishCommands(ctx, e, claim, true, nil, evaldomain.CommandStart, evaldomain.CommandResume); err != nil {
			return false, err
		}
		if e.ControlMode == evaldomain.ControlExternal {
			return reconciled, nil
		}
		admitted, settled, err := s.admitNativeMembers(ctx, claim)
		if err != nil || settled {
			return settled, err
		}
		return len(admitted) > 0 || reconciled, nil
	case evaldomain.StatePausing:
		if e.Outstanding == 0 {
			if err = s.tx(ctx, func(st *evalstore.Store) error {
				return st.Transition(ctx, scope(e), e.ID, claim, evaldomain.StatePausing, evaldomain.StatePaused, e.ObservedTokens, nil)
			}); err != nil {
				return false, err
			}
			return true, s.finishCommands(ctx, e, claim, true, nil, evaldomain.CommandPause)
		}
	case evaldomain.StateSettling:
		if e.Outstanding == 0 {
			if err = s.tx(ctx, func(st *evalstore.Store) error {
				return st.Transition(ctx, scope(e), e.ID, claim, evaldomain.StateSettling, evaldomain.StateFinished, e.ObservedTokens, nil)
			}); err != nil {
				return false, err
			}
			return true, s.finishCommands(ctx, e, claim, true, nil, evaldomain.CommandFinalize)
		}
	case evaldomain.StateCancelling:
		if e.Outstanding == 0 {
			if e.DeletionRequestedAt != nil {
				return s.purge(ctx, e)
			}
			if err = s.tx(ctx, func(st *evalstore.Store) error {
				return st.Transition(ctx, scope(e), e.ID, claim, evaldomain.StateCancelling, evaldomain.StateCancelled, e.ObservedTokens, nil)
			}); err != nil {
				return false, err
			}
			return true, s.finishCommands(ctx, e, claim, true, nil, evaldomain.CommandCancel)
		}
	}
	return reconciled, nil
}

func (s *Service) admitNativeMembers(ctx context.Context, claim evalstore.Claim) ([]evalstore.Member, bool, error) {
	var admitted []evalstore.Member
	settled := false
	err := s.tx(ctx, func(st *evalstore.Store) error {
		current, err := st.GetClaimed(ctx, claim)
		if err != nil {
			return err
		}
		if current.State != evaldomain.StateRunning || current.ControlMode != evaldomain.ControlServer {
			return nil
		}
		capacity := min(current.MaxInFlight-current.Outstanding, membersPerTick)
		limit := max(1, capacity) // An empty candidate set settles even at full capacity.
		members, err := st.NextMembers(ctx, current.OwnerID, current.ID, limit)
		if err != nil {
			return err
		}
		if len(members) == 0 {
			if err := st.Transition(ctx, scope(current), current.ID, claim,
				evaldomain.StateRunning, evaldomain.StateSettling, current.ObservedTokens, nil); err != nil {
				return err
			}
			settled = true
			return nil
		}
		if capacity <= 0 {
			return nil
		}
		plan, err := st.FrozenPlan(ctx, current.OwnerID, current.ID)
		if err != nil {
			return err
		}
		body, _ := jsonBytes(evaldomain.Submission{PlanSHA256: plan.SHA256})
		for _, member := range members {
			identity, err := evaldomain.IdentifyMutation(member.SubmissionKey, "", false, "Submission", body)
			if err != nil {
				return err
			}
			if _, err = st.Admit(ctx, evalstore.Admission{
				Scope:        scope(current),
				ExperimentID: current.ID,
				MemberID:     member.MemberID,
				PlanSHA256:   plan.SHA256,
				Claim:        &claim,
				Mutation:     identity,
			}); err != nil {
				return err
			}
		}
		admitted = members
		return nil
	})
	return admitted, settled, err
}

// Unknown failures retain the pending command for retry. Only a classified
// configuration failure returns the experiment to its editable draft. The
// original cause always reaches the coordinator, even when persistence succeeds.
func (s *Service) preparationFailed(ctx context.Context, e evalstore.Experiment, claim evalstore.Claim, cause error) (bool, error) {
	if ctx.Err() != nil || errors.Is(cause, evalstore.ErrClaimLost) {
		return false, cause
	}
	target, code := evaldomain.StatePreparing, "eval_preparation_unavailable"
	var domain *evaldomain.Error
	if errors.As(cause, &domain) && domain.Status < 500 {
		target, code = evaldomain.StateDraft, domain.Code
	}
	detail := diagnostic(code)
	persistErr := s.tx(ctx, func(st *evalstore.Store) error {
		return st.Transition(ctx, scope(e), e.ID, claim, evaldomain.StatePreparing, target, e.ObservedTokens, detail)
	})
	if persistErr == nil && target == evaldomain.StateDraft {
		persistErr = s.finishCommands(ctx, e, claim, false, detail, evaldomain.CommandPrepare)
	}
	return persistErr == nil, errors.Join(fmt.Errorf("prepare evaluation %s: %w", e.ID, cause), persistErr)
}

// A crash can occur after committing a transition but before acknowledging its
// command. Ready and terminal experiments with pending commands remain claimable.
func (s *Service) recoverCommands(ctx context.Context, e evalstore.Experiment, claim evalstore.Claim) error {
	rows, err := evalstore.NewPostgresStore(s.pool).PendingCommands(ctx, e.OwnerID, e.ID)
	if err != nil {
		return err
	}
	for _, cmd := range rows {
		completion := e.Lifecycle().Completion(cmd.Kind)
		if completion == evaldomain.CommandPending {
			continue
		}
		success := completion == evaldomain.CommandSucceeded

		var detail json.RawMessage
		if !success {
			detail = e.Diagnostic
			if len(detail) == 0 {
				detail = diagnostic("eval_not_ready")
			}
		}
		if err = s.tx(ctx, func(st *evalstore.Store) error {
			return st.CompleteCommand(ctx, scope(e), e.ID, cmd.ID, claim, success, detail)
		}); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) finishCommands(ctx context.Context, e evalstore.Experiment, claim evalstore.Claim, success bool, detail json.RawMessage, kinds ...evaldomain.CommandKind) error {
	rows, err := evalstore.NewPostgresStore(s.pool).PendingCommands(ctx, e.OwnerID, e.ID)
	if err != nil {
		return err
	}
	for _, c := range rows {
		if slices.Contains(kinds, c.Kind) {
			if err = s.tx(ctx, func(st *evalstore.Store) error {
				return st.CompleteCommand(ctx, scope(e), e.ID, c.ID, claim, success, detail)
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Service) purge(ctx context.Context, e evalstore.Experiment) (bool, error) {
	deps, err := evalstore.NewPostgresStore(s.pool).Dependencies(ctx, e.OwnerID, e.ID, "", evaldomain.MaxPageSize)
	if err != nil {
		return false, err
	}
	for _, id := range deps {
		projects := projectstore.NewPostgresStore(s.pool)
		p, err := projects.Get(ctx, e.OwnerID, id)
		if err != nil {
			return false, err
		}
		if p.Lifecycle == projectstore.LifecycleActive {
			_, _, err = projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: e.OwnerID, ProjectID: id, ExpectedRevision: p.Revision})
			return err == nil, err
		}
	}
	if len(deps) > 0 {
		return false, nil
	}
	err = s.tx(ctx, func(st *evalstore.Store) error { return st.Purge(ctx, scope(e), e.ID) })
	return err == nil, err
}
