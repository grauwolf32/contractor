package projectworkflows

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

// These are configuration observations, not proof of upstream weights or inference.
type liveModelSelection struct {
	Workflow         string                        `json:"workflow"`
	Stage            string                        `json:"stage"`
	Agent            string                        `json:"agent"`
	Variant          string                        `json:"variant"`
	ModelPolicy      contracts.ModelPolicyRef      `json:"modelPolicy"`
	Gateway          contracts.LLMGatewayConfigRef `json:"gateway"`
	ModelAliasSHA256 string                        `json:"modelAliasSha256"`
}

type liveResolvedWorker struct {
	evidence liveModelSelection
	config   config.ResolvedConsumerExecutionConfig
}

func liveAliasSHA256(model string) string {
	digest := sha256.Sum256([]byte(model))
	return hex.EncodeToString(digest[:])
}

func copyLiveConfiguration(repositoryRoot, target string, settings liveSettings) ([]liveModelSelection, error) {
	source := filepath.Join(repositoryRoot, "configs")
	if err := filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		destination := filepath.Join(target, relative)
		if entry.Type()&os.ModeSymlink != 0 {
			return errors.New("configuration contains a symbolic link")
		}
		if entry.IsDir() {
			return os.MkdirAll(destination, 0o700)
		}
		if !entry.Type().IsRegular() {
			return errors.New("configuration contains a non-regular entry")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(destination, data, 0o600)
	}); err != nil {
		return nil, errors.New("copy live configuration failed")
	}
	snapshot, err := config.Load(target, config.MVPDescriptors())
	if err != nil {
		return nil, errors.New("live configuration cannot be resolved")
	}
	workers, err := resolveLiveWorkers(snapshot, settings.workflows)
	if err != nil {
		return nil, err
	}
	policies, gateways := map[string]bool{}, map[string]bool{}
	for _, worker := range workers {
		ref := worker.config.ModelPolicy.Ref
		policies[ref.PolicyID+"@"+ref.Version] = true
		gateway := worker.config.LLMGateway.Ref
		gateways[gateway.GatewayID+"@"+gateway.Version] = true
	}
	// A policy reused by another model role cannot safely be rewritten in place.
	// Fail explicitly instead of silently changing a Planner or summarizer.
	for _, workflow := range snapshot.Workflows() {
		for _, stage := range workflow.Stages {
			for _, selection := range liveStageVariants(stage) {
				if selection.Planner != nil {
					ref := selection.Planner.ModelPolicy.Ref
					if policies[ref.PolicyID+"@"+ref.Version] {
						return nil, errors.New("selected Worker policy is shared with a Planner")
					}
				}
			}
			for _, agent := range stage.Agents {
				if agent.Template.Summarizer != nil {
					ref := agent.Template.Summarizer.ModelPolicy.Ref
					if policies[ref.PolicyID+"@"+ref.Version] {
						return nil, errors.New("selected Worker policy is shared with a summarizer")
					}
				}
			}
		}
	}
	if err := rewriteLiveManifests(target, "model-policies", policies, "model", settings.model); err != nil {
		return nil, err
	}
	if err := rewriteLiveManifests(target, "llm-gateways", gateways, "url", settings.gatewayURL); err != nil {
		return nil, err
	}
	// Recompute all identities and effective selections with the same loader the Server uses.
	snapshot, err = config.Load(target, config.MVPDescriptors())
	if err != nil {
		return nil, errors.New("overridden live configuration cannot be resolved")
	}
	workers, err = resolveLiveWorkers(snapshot, settings.workflows)
	if err != nil {
		return nil, err
	}
	result := make([]liveModelSelection, 0, len(workers))
	for _, worker := range workers {
		if worker.config.ModelPolicy.Model != settings.model || worker.config.LLMGateway.URL != settings.gatewayURL {
			return nil, errors.New("live Worker override did not reach its effective route")
		}
		result = append(result, worker.evidence)
	}
	return result, nil
}

func liveStageVariants(stage config.ResolvedStage) map[string]config.ResolvedStageExecutionConfig {
	variants := map[string]config.ResolvedStageExecutionConfig{"base": stage.ExecutionConfig}
	for outcome, action := range map[string]config.TransitionAction{
		"failed": stage.On.Failed, "interrupted": stage.On.Interrupted,
	} {
		if action.Escalate != nil {
			variants[outcome+"_escalation"] = action.Escalate.ExecutionConfig.Effective
		}
	}
	return variants
}

func resolveLiveWorkers(snapshot *config.Snapshot, selected []string) ([]liveResolvedWorker, error) {
	var result []liveResolvedWorker
	for _, name := range selected {
		workflow, err := snapshot.Workflow(name)
		if err != nil {
			return nil, errors.New("selected live Workflow cannot be resolved")
		}
		for stageName, stage := range workflow.Stages {
			for variant, execution := range liveStageVariants(stage) {
				for agent, selection := range execution.Agents {
					if selection.ModelPolicy.IsZero() || selection.LLMGateway == nil {
						return nil, errors.New("selected live Worker has no model policy or Gateway route")
					}
					result = append(result, liveResolvedWorker{
						evidence: liveModelSelection{
							Workflow: name, Stage: stageName, Agent: agent, Variant: variant,
							ModelPolicy: selection.ModelPolicy.Ref, Gateway: selection.LLMGateway.Ref,
							ModelAliasSHA256: liveAliasSHA256(selection.ModelPolicy.Model),
						},
						config: selection,
					})
				}
			}
		}
	}
	if len(result) == 0 {
		return nil, errors.New("live configuration selects no modeled Workers")
	}
	sort.Slice(result, func(i, j int) bool {
		a, b := result[i].evidence, result[j].evidence
		return a.Workflow+"\x00"+a.Stage+"\x00"+a.Variant+"\x00"+a.Agent < b.Workflow+"\x00"+b.Stage+"\x00"+b.Variant+"\x00"+b.Agent
	})
	return result, nil
}

// Match exact manifest identities, not filenames or the former model alias.
// Only selected spec scalar values change; all limits and other settings survive.
func rewriteLiveManifests(root, directory string, selected map[string]bool, field, value string) error {
	remaining := make(map[string]bool, len(selected))
	for key := range selected {
		remaining[key] = true
	}
	err := filepath.WalkDir(filepath.Join(root, directory), func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || (filepath.Ext(path) != ".yaml" && filepath.Ext(path) != ".yml") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var document yaml.Node
		if err := yaml.Unmarshal(data, &document); err != nil {
			return err
		}
		var identity struct {
			Metadata struct {
				Name    string `yaml:"name"`
				Version string `yaml:"version"`
			} `yaml:"metadata"`
		}
		if err := document.Decode(&identity); err != nil {
			return err
		}
		key := identity.Metadata.Name + "@" + identity.Metadata.Version
		if !selected[key] {
			return nil
		}
		spec := liveYAMLField(document.Content[0], "spec")
		scalar := liveYAMLField(spec, field)
		if scalar == nil || scalar.Kind != yaml.ScalarNode {
			return errors.New("selected live configuration field is absent")
		}
		scalar.SetString(value)
		encoded, err := yaml.Marshal(&document)
		if err != nil {
			return err
		}
		if err := os.WriteFile(path, encoded, 0o600); err != nil {
			return err
		}
		delete(remaining, key)
		return nil
	})
	if err != nil || len(remaining) != 0 {
		return errors.New("selected live configuration manifest could not be overridden")
	}
	return nil
}

func liveYAMLField(node *yaml.Node, name string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i < len(node.Content); i += 2 {
		if node.Content[i].Value == name {
			return node.Content[i+1]
		}
	}
	return nil
}
