package evalservice

import (
	"context"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Catalog interface {
	ResolveRunWorkflow(context.Context, string, config.ExecutionConfigPatch, config.CredentialLookup) (config.ResolvedWorkflow, error)
	AuditProfile(string) (config.ResolvedAuditProfile, error)
}
type CredentialBarrier interface {
	WithRunCreation(context.Context, func() error) error
}
type Resolver struct {
	Pool        *pgxpool.Pool
	Catalog     Catalog
	Credentials runtimeconfig.TransactionLLMCredentialLookupFactory
	Barrier     CredentialBarrier
}
type BindingSnapshot struct {
	Workflow  *config.ResolvedWorkflow      `json:"workflow,omitempty"`
	Audit     *config.ResolvedAuditProfile  `json:"audit,omitempty"`
	Runtime   runtimeconfig.RunSnapshot     `json:"runtime"`
	Skills    []contracts.RunSkillSnapshot  `json:"skills"`
	Standards []auditstandards.ExactPackage `json:"standards"`
}

// Resolve obtains safe configuration and exact artifact metadata. It makes no
// Run, Audit, target, Artifact write, model call, or external producer callback.
// Shared catalog locks are short-lived; dispatch checks these frozen pins again.
func (r *Resolver) Resolve(ctx context.Context, owner string, v evaldomain.Variant, cases []evaldomain.Case) (Preflight, error) {
	out := Preflight{Pins: map[string]Pin{}, Cases: map[string]Eligibility{}, Capabilities: []string{"input.artifact@1", "output.artifact@1"}}
	if r.Pool == nil || r.Catalog == nil || r.Credentials == nil || r.Barrier == nil {
		return out, errors.New("eval preflight dependencies are incomplete")
	}
	if b, err := jsonBytes(v); err != nil {
		return out, err
	} else if err = evaldomain.Validate("Variant", b); err != nil {
		return out, err
	}
	var resolvedStandards []auditstandards.ResolvedPackage
	snapshot := BindingSnapshot{Skills: []contracts.RunSkillSnapshot{}, Standards: []auditstandards.ExactPackage{}}
	err := r.Barrier.WithRunCreation(ctx, func() error {
		return pg.InTx(ctx, r.Pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, r.Credentials)
			if err != nil {
				return err
			}
			catalogLookup := &preflightCredentialLookup{CredentialLookup: lookup}
			artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
			workflows := map[string]config.ResolvedWorkflow{}
			slots := map[string]config.ArtifactSlot{}
			modelFree := true
			if v.Kind == "workflow" {
				workflow, err := r.Catalog.ResolveRunWorkflow(ctx, v.Selector, v.ExecutionConfig, catalogLookup)
				if err != nil {
					return catalogLookup.catalogError(err)
				}
				snapshot.Workflow = &workflow
				workflows["workflow"] = workflow
				slots = workflow.Inputs
				out.Capabilities = append(out.Capabilities, "workflow.run@1")
			} else {
				patch, err := jsonBytes(v.ExecutionConfig)
				if err != nil {
					return err
				}
				if string(patch) != "{}" {
					return evaldomain.Failure("eval_not_ready")
				}
				profile, err := r.Catalog.AuditProfile(v.Selector)
				if err != nil {
					return errors.Join(evaldomain.Failure("eval_not_ready"), err)
				}
				if !auditservice.ProfileCompatibility(profile).ServerCompatible {
					return evaldomain.Failure("eval_not_ready")
				}
				snapshot.Audit = &profile
				out.Capabilities = append(out.Capabilities, "audit.run@1")
				for name, slot := range profile.Inputs {
					slots[name] = config.ArtifactSlot{Required: slot.Required, MediaTypes: slot.MediaTypes}
				}
				for role, binding := range profile.Workflows {
					workflows[role] = binding.Workflow
				}
				catalog, err := auditstandards.NewCatalog(artifactService)
				if err != nil {
					return err
				}
				for _, standard := range profile.Standards {
					resolved, err := catalog.Resolve(ctx, owner, auditstandards.Reference{Scheme: standard.Scheme, Version: standard.Version})
					if err != nil {
						return preflightDependencyError(err)
					}
					snapshot.Standards = append(snapshot.Standards, resolved.Source)
					resolvedStandards = append(resolvedStandards, resolved)
				}
			}
			skillRefs := map[string]contracts.ArtifactRef{}
			var skillSets [][]string
			for _, workflow := range workflows {
				if err = config.ValidateResolvedWorkflowCredentials(ctx, workflow, catalogLookup); err != nil {
					return catalogLookup.catalogError(err)
				}
				refs, err := config.WorkflowSkillRefs(workflow)
				if err != nil {
					return errors.Join(evaldomain.Failure("eval_not_ready"), err)
				}
				for _, ref := range refs {
					skillRefs[ref.Name] = ref
				}
				skillSets = append(skillSets, config.WorkflowSkillSets(workflow)...)
				for _, stage := range workflow.Stages {
					for _, agent := range stage.Agents {
						modelFree = modelFree && agent.Template.IsToolWorker()
					}
				}
			}
			if v.Kind == "audit" {
				modelFree = false
			}
			runtime, err := runtimeconfig.PinRunSnapshot(ctx, tx, v.RuntimeLabels, credentials.NewRuntimeCredentialRepository(tx), lookup, modelFree)
			if err != nil {
				return preflightDependencyError(err)
			}
			snapshot.Runtime = runtime
			refs := make([]contracts.ArtifactRef, 0, len(skillRefs))
			for _, ref := range skillRefs {
				refs = append(refs, ref)
			}
			sort.Slice(refs, func(i, j int) bool { return refs[i].Name < refs[j].Name })
			if len(refs) > 0 {
				catalog, err := agentskills.NewCatalog(artifactService)
				if err != nil {
					return err
				}
				snapshot.Skills, err = catalog.SelectRunSources(ctx, owner, refs)
				if err != nil {
					return preflightDependencyError(err)
				}
				if err = agentskills.ValidateSelectedLimits(snapshot.Skills, skillSets); err != nil {
					return errors.Join(evaldomain.Failure("eval_not_ready"), err)
				}
				for _, skill := range snapshot.Skills {
					if skill.Source == nil {
						return evaldomain.Failure("eval_not_ready")
					}
				}
			}
			for _, slot := range slots {
				for _, media := range slot.MediaTypes {
					if media == "application/zip" {
						out.Capabilities = append(out.Capabilities, "source.archive@1")
					}
				}
			}
			out.Capabilities = unique(out.Capabilities)
			for _, c := range cases {
				for _, ref := range c.Inputs {
					if err = verifyArtifact(ctx, tx, artifactService, owner, ref); err != nil {
						return err
					}
				}
				out.Cases[c.ID] = caseEligibility(c, v, snapshot, out.Capabilities)
				if snapshot.Audit != nil && out.Cases[c.ID].State == "eligible" {
					mapped, err := MapInputs(c, v)
					if err != nil {
						return err
					}
					payloads := map[string]artifacts.ReadResult{}
					for slot, ref := range mapped {
						store, err := artifactScope(artifactService, ref)
						if err != nil {
							return err
						}
						read, err := store.Read(ctx, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision})
						if err != nil {
							return preflightDependencyError(err)
						}
						if evaldomain.Digest(read.Payload.Data) != ref.SHA256 {
							return evaldomain.Failure("eval_pin_mismatch")
						}
						payloads[slot] = read
					}
					params := MapParameters(c, v)
					scope := auditservice.Scope{Objective: c.Task.Objective, Target: params["target"], AuthorizationScope: params["authorizationScope"]}
					if objective, ok := params["objective"]; ok {
						scope.Objective = objective
					}
					if err = auditservice.ValidateInputPreview(*snapshot.Audit, scope, payloads, resolvedStandards); err != nil {
						out.Cases[c.ID] = unavailable("The Audit input or inventory is not supported by the selected profile.")
					}
				}

			}
			for dimension, value := range bindingPinValues(snapshot, workflows) {
				digest, err := hashJSON(value)
				if err != nil {
					return err
				}
				out.Pins[dimension] = observedPin(digest)
			}
			// A selected model policy does not prove a provider's deployed revision.
			out.Pins["model-revision"] = Pin{nil, "unavailable"}
			out.Pins["runtime-build"] = Pin{nil, "unavailable"}
			out.Pins["tool-docstrings"] = Pin{nil, "unavailable"}
			out.Snapshot, err = jsonBytes(snapshot)
			return err
		})
	})
	return out, err
}
func verifyArtifact(ctx context.Context, db pg.DBTX, service *artifacts.Service, owner string, ref evaldomain.Artifact) error {
	var store artifacts.ScopedStore
	var err error
	switch ref.Scope {
	case "user":
		if ref.ScopeID != owner {
			return evaldomain.Failure("eval_not_found")
		}
		store, err = service.User(owner)
	case "project":
		if _, err = projectstore.NewPostgresStore(db).Get(ctx, owner, ref.ScopeID); err != nil {
			return preflightDependencyError(err)
		}
		store, err = service.Project(ref.ScopeID)
	default:
		return evaldomain.Failure("eval_invalid")
	}
	if err != nil {
		return err
	}
	metadata, err := store.Metadata(ctx, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision})
	if err != nil {
		return preflightDependencyError(err)
	}
	if metadata.Digest != ref.SHA256 || metadata.MediaType != ref.MediaType || metadata.Size != ref.SizeBytes {
		return evaldomain.Failure("eval_pin_mismatch")
	}
	return nil
}
func unique(in []string) []string {
	sort.Strings(in)
	out := []string{}
	for _, v := range in {
		if len(out) == 0 || out[len(out)-1] != v {
			out = append(out, v)
		}
	}
	return out
}
func unavailable(reason string) Eligibility {
	return Eligibility{State: "unsupported", Reason: &reason}
}
func caseEligibility(c evaldomain.Case, v evaldomain.Variant, s BindingSnapshot, capabilities []string) Eligibility {
	known := map[string]bool{}
	for _, cap := range capabilities {
		known[cap] = true
	}
	for _, required := range c.Requires {
		if !known[required] {
			return unavailable("The binding does not declare a required capability.")
		}
	}
	inputs, err := MapInputs(c, v)
	if err != nil {
		return unavailable("The input mapping is incompatible with this case.")
	}
	slots := map[string]config.ArtifactSlot{}
	outputs := map[string]config.ArtifactSlot{}
	if s.Workflow != nil {
		slots = s.Workflow.Inputs
		outputs = s.Workflow.Outputs
		params := MapParameters(c, v)
		for name := range params {
			if _, ok := s.Workflow.Parameters[name]; !ok {
				return unavailable("The case supplies an unknown Workflow parameter.")
			}
		}
		for name, slot := range s.Workflow.Parameters {
			if _, ok := params[name]; slot.Required && !ok {
				return unavailable("A required Workflow parameter is missing.")
			}
		}
	} else {
		for name, slot := range s.Audit.Inputs {
			slots[name] = config.ArtifactSlot{Required: slot.Required, MediaTypes: slot.MediaTypes}
		}
		outputs["report"] = config.ArtifactSlot{MediaTypes: []string{"application/json", "text/markdown"}}
		for name := range MapParameters(c, v) {
			if name != "objective" && name != "target" && name != "authorizationScope" {
				return unavailable("The Audit scope has an unsupported parameter.")
			}
		}
	}
	for name, ref := range inputs {
		slot, ok := slots[name]
		if !ok || !contains(slot.MediaTypes, ref.MediaType) {
			return unavailable("An input slot or media type is incompatible.")
		}
	}
	for name, slot := range slots {
		if _, ok := inputs[name]; slot.Required && !ok {
			return unavailable("A required input is missing.")
		}
	}
	for role, contract := range c.Outputs {
		name := role
		if mapped, ok := v.OutputMapping[role]; ok {
			name = mapped
		}
		slot, ok := outputs[name]
		if !ok && contract.Required {
			return unavailable("A required output role is unavailable.")
		}
		if ok {
			compatible := false
			for _, media := range contract.MediaTypes {
				compatible = compatible || contains(slot.MediaTypes, media)
			}
			if !compatible {
				return unavailable("An output media type is incompatible.")
			}
		}
	}
	return Eligibility{State: "eligible"}
}

// Mappings use case role -> executable slot. Omitted entries keep their name.
func MapInputs(c evaldomain.Case, v evaldomain.Variant) (map[string]evaldomain.Artifact, error) {
	out := map[string]evaldomain.Artifact{}
	for role := range v.InputMapping {
		if _, ok := c.Inputs[role]; !ok {
			return nil, evaldomain.Failure("eval_invalid")
		}
	}
	for role, ref := range c.Inputs {
		slot := role
		if mapped, ok := v.InputMapping[role]; ok {
			slot = mapped
		}
		if _, ok := out[slot]; ok {
			return nil, evaldomain.Failure("eval_invalid")
		}
		out[slot] = ref
	}
	return out, nil
}
func MapParameters(c evaldomain.Case, v evaldomain.Variant) map[string]string {
	out := map[string]string{}
	for k, value := range c.Task.Parameters {
		out[k] = value
	}
	for k, value := range v.Parameters {
		if value == "$task.objective" {
			value = c.Task.Objective
		}
		out[k] = value
	}
	return out
}
func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
func bindingPinValues(s BindingSnapshot, workflows map[string]config.ResolvedWorkflow) map[string]any {
	instructions := map[string]any{}
	tools := map[string]any{}
	models := map[string]any{}
	policies := map[string]any{}
	for role, w := range workflows {
		for name, stage := range w.Stages {
			key := role + "/" + name
			instructions[key] = map[string]any{"instructions": stage.Instructions, "objective": stage.Objective}
			models[key] = stage.ExecutionConfig
			policies[key] = stage.On
			for agent, binding := range stage.Agents {
				k := key + "/" + agent
				instructions[k] = binding.Template.Instructions
				tools[k] = map[string]any{"tools": binding.Template.Toolsets, "execution": binding.Template.Execution, "sandbox": binding.Template.SandboxProfile, "runtime": binding.Template.Runtime}
				models[k] = binding.Template.Summarizer
			}
		}
	}
	out := map[string]any{"instructions": instructions, "tools": tools, "models": models, "sampling": models, "skills": s.Skills, "runtime-config": s.Runtime, "execution": policies, "standards": s.Standards}
	if s.Audit != nil {
		out["inventory"] = s.Audit.Inventory
		out["audit-execution"] = s.Audit.Execution
		out["audit-interaction"] = s.Audit.Interaction
	}
	return out
}

// artifactScope follows a preceding owner check; it must not be exposed as an
// authorization shortcut to a public request handler.
func artifactScope(service *artifacts.Service, ref evaldomain.Artifact) (artifacts.ScopedStore, error) {
	if ref.Scope == "user" {
		return service.User(ref.ScopeID)
	}
	if ref.Scope == "project" {
		return service.Project(ref.ScopeID)
	}
	return artifacts.ScopedStore{}, evaldomain.Failure("eval_invalid")
}
