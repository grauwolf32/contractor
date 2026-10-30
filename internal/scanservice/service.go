// Package scanservice admits owner-bound immutable Nuclei sweeps. It creates
// no scanner work directly: the controller submits ordinary Project Runs.
package scanservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A sweep carries at most 32 parameters, each name and value bounded in UTF-8
// bytes as the public schema declares (LIM-SCN-94).
const (
	MaximumParameterNameBytes  = 128
	MaximumParameterValueBytes = 2048
)

type Request struct {
	ProjectIDs  []string               `json:"projectIds,omitempty"`
	AllProjects bool                   `json:"allProjects,omitempty"`
	Workflow    string                 `json:"workflow"`
	Parameters  map[string]string      `json:"parameters"`
	Rulepack    *contracts.ArtifactRef `json:"rulepack,omitempty"`
}

type RuntimePinner func(context.Context, pgx.Tx) (runtimeconfig.RunSnapshot, error)
type Drainer interface {
	Drain(context.Context, string, func() error) error
}

type Options struct {
	Logger      *slog.Logger
	Drainer     Drainer
	Pool        *pgxpool.Pool
	Workflows   runservice.WorkflowResolver
	Credentials resolved.CredentialLookup
	PinRuntime  RuntimePinner
	NewID       func(string) (string, error)
}
type Service struct{ options Options }

func New(o Options) (*Service, error) {
	if o.Pool == nil || o.Workflows == nil || o.PinRuntime == nil || o.NewID == nil || o.Drainer == nil || o.Credentials == nil || o.Logger == nil {
		return nil, scanstore.ErrInvalid
	}
	return &Service{options: o}, nil
}

func normalize(r Request) (Request, error) {
	r.ProjectIDs = slices.Clone(r.ProjectIDs)
	slices.Sort(r.ProjectIDs)
	if r.AllProjects == (len(r.ProjectIDs) > 0) || len(r.ProjectIDs) > scanstore.MaxProjects {
		return Request{}, scanstore.ErrInvalid
	}
	for i, id := range r.ProjectIDs {
		if !contracts.ValidResourceID(id) || (i > 0 && id == r.ProjectIDs[i-1]) {
			return Request{}, scanstore.ErrInvalid
		}
	}
	if r.Workflow == "" {
		r.Workflow = "nuclei-web@1"
	}
	if !strings.Contains(r.Workflow, "@") || len(r.Workflow) > 256 {
		return Request{}, scanstore.ErrInvalid
	}
	r.Parameters = maps.Clone(r.Parameters)
	if r.Parameters == nil {
		r.Parameters = map[string]string{}
	}
	if _, ok := r.Parameters["target"]; ok {
		return Request{}, scanstore.ErrInvalid
	}
	if len(r.Parameters) > 32 {
		return Request{}, scanstore.ErrInvalid
	}
	for key, value := range r.Parameters {
		if len(key) > MaximumParameterNameBytes || len(value) > MaximumParameterValueBytes {
			return Request{}, scanstore.ErrInvalid
		}
	}
	if r.Rulepack != nil {
		ref := r.Rulepack.Clone()
		if ref.ValidateExact() != nil || ref.Namespace == artifactpolicy.RunSystemNamespace {
			return Request{}, scanstore.ErrInvalid
		}
		r.Rulepack = &ref
	}
	return r, nil
}

// Create replays the receipt before resolving mutable configuration, Projects
// or source bindings. Purged sweeps retain the original receipt, so replay can
// never turn into new traffic under the same key.
func (s *Service) Create(ctx context.Context, owner, key string, request Request) (scanstore.Sweep, bool, error) {
	request, err := normalize(request)
	if err != nil || !contracts.ValidResourceID(owner) || !contracts.ValidResourceID(key) || len(key) > 128 {
		return scanstore.Sweep{}, false, scanstore.ErrInvalid
	}
	requestDigest, err := contentdigest.CanonicalJSON(request)
	if err != nil {
		return scanstore.Sweep{}, false, err
	}
	id, err := s.options.NewID("sweep_")
	if err != nil {
		return scanstore.Sweep{}, false, err
	}
	var result scanstore.Sweep
	var replayed bool
	err = persistencepostgres.InTxWithRetry(ctx, s.options.Pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		receipt, replay, err := persistencepostgres.KeyedMutation(ctx, tx, persistencepostgres.ScanSweepReceiptKey(owner, key), requestDigest, func() (persistencepostgres.Receipt, error) {
			workflow, err := s.options.Workflows.ResolveRunWorkflow(ctx, request.Workflow, resolved.ExecutionConfigPatch{}, s.options.Credentials)
			if err != nil {
				return persistencepostgres.Receipt{}, fmt.Errorf("%w: invalid sweep Workflow", scanstore.ErrInvalid)
			}
			if err := validateWorkflow(workflow, request); err != nil {
				return persistencepostgres.Receipt{}, err
			}
			runtime, err := s.options.PinRuntime(ctx, tx)
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			if runtime.Validate() != nil {
				return persistencepostgres.Receipt{}, scanstore.ErrInvalid
			}
			snapshot := scanstore.Snapshot{SchemaVersion: 1, OwnerID: owner, Workflow: workflow, Runtime: runtime}
			snapshot.WorkflowDigest, err = contentdigest.JSON(workflow)
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			if request.Rulepack != nil {
				a := artifacts.NewService(artifacts.NewPostgresRepository(tx))
				user, _ := a.User(owner)
				source, err := user.Metadata(ctx, *request.Rulepack)
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				if !contracts.AcceptsMediaType(workflow.Inputs["rulepack"].MediaTypes, source.MediaType) || !scan.ValidNucleiRulepackMetadata(source.MediaType, source.Size) {
					return persistencepostgres.Receipt{}, scanstore.ErrInvalid
				}
				fork, err := a.RetainSweepInput(ctx, owner, id, source.Ref)
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				snapshot.Rulepack = &scanstore.InputPin{Source: source.Exact(), Retained: fork.TargetRef}
			}
			projects, err := captureProjects(ctx, tx, owner, request)
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			if len(projects) < 1 || len(projects) > scanstore.MaxProjects {
				return persistencepostgres.Receipt{}, scanstore.ErrInvalid
			}
			members := make([]scanstore.Member, 0, len(projects))
			for i, project := range projects {
				params := maps.Clone(request.Parameters)
				params["target"] = project.Target
				if project.Reason == "" {
					if err := projectArguments(workflow, params); err != nil {
						project.Reason = "invalid_target"
					}
				}
				child := scanstore.Child{Workflow: request.Workflow, Parameters: params, Inputs: map[string]contracts.ArtifactRef{}, Labels: map[string]string{"scan-sweep": id}}
				if snapshot.Rulepack != nil {
					child.Inputs["rulepack"] = snapshot.Rulepack.Source.Ref
				}
				encoded, err := strictjson.Canonical(child)
				if err != nil || len(encoded) > scanstore.MaxChildBytes {
					return persistencepostgres.Receipt{}, scanstore.ErrInvalid
				}
				childKey, err := s.options.NewID("sweep_child_")
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				digest, err := runservice.PublicRequestDigest(runservice.PublicCreateParams{ProjectID: &project.ProjectID, Workflow: child.Workflow, Parameters: child.Parameters, Inputs: child.Inputs, MetadataLabels: child.Labels})
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				state := "pending"
				if project.Reason != "" {
					state = "skipped"
				}
				members = append(members, scanstore.Member{Ordinal: i, ProjectID: project.ProjectID, Target: project.Target, State: state, Reason: project.Reason, Child: encoded, Key: childKey, Digest: digest})
			}
			encoded, err := strictjson.Canonical(snapshot)
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			// Bind membership and every immutable key/body to the closure digest.
			immutable := struct {
				Snapshot scanstore.Snapshot `json:"snapshot"`
				Members  []scanstore.Member `json:"members"`
			}{snapshot, members}
			// Child/key fields are intentionally absent from public Member JSON.
			// Include their bytes explicitly in the canonical digest below.
			digests := make([]any, 0, len(members))
			for _, m := range members {
				digests = append(digests, []any{m.ProjectID, m.Target, m.Key, m.Digest, string(m.Child), m.Reason})
			}
			digest, err := contentdigest.CanonicalJSON([]any{immutable.Snapshot, digests})
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			if err := scanstore.New(tx).Insert(ctx, id, owner, request.Workflow, digest, encoded, members); err != nil {
				return persistencepostgres.Receipt{}, err
			}
			response, err := strictjson.Canonical(map[string]string{"sweepId": id})
			if err != nil {
				return persistencepostgres.Receipt{}, err
			}
			return persistencepostgres.Receipt{Operation: "scan-sweep.create", ResourceID: id, Response: response, OwnerID: owner}, nil
		})
		if errors.Is(err, persistencepostgres.ErrReceiptMismatch) {
			return scanstore.ErrIdempotency
		}
		if err != nil {
			return err
		}
		replayed = replay
		result, err = scanstore.New(tx).Get(ctx, owner, receipt.ResourceID)
		return err
	})
	return result, replayed, err
}

func validateWorkflow(w resolved.ResolvedWorkflow, r Request) error {
	if r.Workflow != w.Ref.Name+"@"+w.Ref.Version {
		return scanstore.ErrInvalid
	}
	if !resolved.WorkflowToolWorkersOnly(w) || len(w.Stages) != 1 || len(w.Inputs) > 1 {
		return scanstore.ErrInvalid
	}
	stage := w.Stages[w.EntryStage]
	if stage.Planner.Selector() != resolved.PassthroughPlanner || len(stage.Agents) != 1 || stage.ScanPlan != nil {
		return scanstore.ErrInvalid
	}
	for _, agent := range stage.Agents {
		if !agent.Template.IsToolWorker() || agent.Template.Execution == nil || agent.Template.Execution.Tool != "scan_nuclei" || len(agent.Template.Skills) > 0 {
			return scanstore.ErrInvalid
		}
		if r.Rulepack == nil {
			pin := agent.Template.Execution.Arguments["templates_digest"]
			value, _ := pin.Value.(string)
			if pin.Source != "literal" || !contentdigest.Valid(value) {
				return scanstore.ErrInvalid
			}
		}
	}
	for name, slot := range w.Inputs {
		if name != "rulepack" || slot.SourceScope != "user" || slot.Required {
			return scanstore.ErrInvalid
		}
	}
	if r.Rulepack != nil {
		if _, ok := w.Inputs["rulepack"]; !ok {
			return scanstore.ErrInvalid
		}
	}
	params := maps.Clone(r.Parameters)
	params["target"] = "https://sweep-validation.invalid/"
	return projectArguments(w, params)
}

func projectArguments(w resolved.ResolvedWorkflow, params map[string]string) error {
	for name := range params {
		if _, ok := w.Parameters[name]; !ok {
			return scanstore.ErrInvalid
		}
	}
	for name, slot := range w.Parameters {
		if _, ok := params[name]; slot.Required && !ok {
			return scanstore.ErrInvalid
		}
	}
	for _, agent := range w.Stages[w.EntryStage].Agents {
		args := map[string]any{}
		for name, binding := range agent.Template.Execution.Arguments {
			switch binding.Source {
			case "literal":
				args[name] = binding.Value
			case "parameter":
				value, err := binding.ResolveParameter(params)
				if err != nil {
					return scanstore.ErrInvalid
				}
				args[name] = value
				if _, ok := params[binding.Name]; !ok {
					switch v := value.(type) {
					case string:
						params[binding.Name] = v
					case int64:
						params[binding.Name] = strconv.FormatInt(v, 10)
					case int:
						params[binding.Name] = strconv.Itoa(v)
					case float64:
						params[binding.Name] = strconv.FormatFloat(v, 'f', -1, 64)
					}
				}
			}
		}
		if scan.ValidateNucleiArguments(args) != nil {
			return scanstore.ErrInvalid
		}
	}
	return nil
}

func (s *Service) Get(ctx context.Context, owner, id string) (scanstore.Sweep, error) {
	return scanstore.New(s.options.Pool).Get(ctx, owner, id)
}
func (s *Service) List(ctx context.Context, owner, after string, limit int) ([]scanstore.Sweep, error) {
	return scanstore.New(s.options.Pool).List(ctx, owner, after, limit)
}
func (s *Service) Members(ctx context.Context, owner, id string, after, limit int) ([]scanstore.Member, error) {
	return scanstore.New(s.options.Pool).Members(ctx, owner, id, after, limit)
}
func (s *Service) Cancel(ctx context.Context, owner, id string) (scanstore.Sweep, error) {
	return scanstore.New(s.options.Pool).Cancel(ctx, owner, id)
}
func (s *Service) Purge(ctx context.Context, owner, id string) error {
	sweep, err := s.Get(ctx, owner, id)
	if err != nil {
		return err
	}
	if sweep.State != "completed" && sweep.State != "cancelled" {
		return scanstore.ErrConflict
	}
	purge := func() error {
		return persistencepostgres.InTxWithRetry(ctx, s.options.Pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if err := scanstore.New(tx).DeleteTerminal(ctx, owner, id); err != nil {
				return err
			}
			purger, err := artifacts.NewPostgresPurger(tx)
			if err != nil {
				return err
			}
			return purger.PurgeSweepInput(ctx, owner, id)
		})
	}
	return s.options.Drainer.Drain(ctx, id, purge)
}

// UnmarshalJSON rejects explicit nulls as well as duplicate, alias and unknown
// keys. Omitting an optional selection differs from supplying a malformed one.
func (r *Request) UnmarshalJSON(data []byte) error {
	type plain Request
	var decoded plain
	if err := strictjson.Decode(data, &decoded); err != nil {
		return err
	}
	var members map[string]json.RawMessage
	if err := strictjson.Decode(data, &members); err != nil {
		return err
	}
	if _, present := members["workflow"]; present && decoded.Workflow == "" {
		return scanstore.ErrInvalid
	}
	for _, value := range members {
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return scanstore.ErrInvalid
		}
	}
	*r = Request(decoded)
	return nil
}
