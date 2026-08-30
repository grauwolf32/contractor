package config

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

type loader struct {
	root             string
	descriptors      Descriptors
	instructions     map[string]contracts.ResolvedInstructions
	policies         map[string]contracts.ResolvedModelPolicy
	gateways         map[string]contracts.ResolvedLLMGatewayConfig
	executionConfigs map[string]ResolvedExecutionConfigProfile
	templates        map[string]contracts.ResolvedAgentTemplate
	workflows        map[string]ResolvedWorkflow
}

type manifestFile struct {
	absolute string
	relative string
}

// Load validates the complete configuration root and publishes one immutable
// snapshot. On any error the returned Snapshot is nil.
func Load(root string, descriptors Descriptors) (*Snapshot, error) {
	normalizedDescriptors, err := normalizeDescriptors(descriptors)
	if err != nil {
		return nil, err
	}
	resolvedRoot, err := resolveRoot(root)
	if err != nil {
		return nil, err
	}

	current := &loader{
		root:             resolvedRoot,
		descriptors:      normalizedDescriptors,
		instructions:     make(map[string]contracts.ResolvedInstructions),
		policies:         make(map[string]contracts.ResolvedModelPolicy),
		gateways:         make(map[string]contracts.ResolvedLLMGatewayConfig),
		executionConfigs: make(map[string]ResolvedExecutionConfigProfile),
		templates:        make(map[string]contracts.ResolvedAgentTemplate),
		workflows:        make(map[string]ResolvedWorkflow),
	}
	if err := current.loadLLMGatewayConfigs(); err != nil {
		return nil, err
	}
	if err := current.loadModelPolicies(); err != nil {
		return nil, err
	}
	if err := current.loadExecutionConfigs(); err != nil {
		return nil, err
	}
	if err := current.loadAgentTemplates(); err != nil {
		return nil, err
	}
	if err := current.loadWorkflows(); err != nil {
		return nil, err
	}
	return newSnapshot(
		current.workflows, current.templates, current.policies, current.gateways,
		current.executionConfigs, current.instructions,
	), nil
}

func resolveRoot(root string) (string, error) {
	if strings.TrimSpace(root) == "" {
		return "", fmt.Errorf("configuration root is required")
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve configuration root %q: %w", root, err)
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", fmt.Errorf("resolve configuration root %q: %w", root, err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("stat configuration root %q: %w", root, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("configuration root %q is not a directory", root)
	}
	return resolved, nil
}

func (l *loader) discover(subtree string) ([]manifestFile, error) {
	base := filepath.Join(l.root, subtree)
	info, err := os.Stat(base)
	if err != nil {
		return nil, fmt.Errorf("configuration subtree %s: %w", subtree, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("configuration subtree %s is not a directory", subtree)
	}

	var result []manifestFile
	err = filepath.WalkDir(base, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			return nil
		}
		if !entry.Type().IsRegular() {
			return nil
		}
		relative, relErr := filepath.Rel(l.root, path)
		if relErr != nil {
			return relErr
		}
		result = append(result, manifestFile{absolute: path, relative: filepath.ToSlash(relative)})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("discover %s manifests: %w", subtree, err)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].relative < result[j].relative })
	return result, nil
}

func decodeOne[T any](file manifestFile) (T, error) {
	var zero T
	data, err := os.ReadFile(file.absolute)
	if err != nil {
		return zero, fmt.Errorf("%s: read manifest: %w", file.relative, err)
	}
	var documents []T
	if err := yaml.Load(
		data,
		&documents,
		yaml.WithAllDocuments(),
		yaml.WithKnownFields(),
		yaml.WithUniqueKeys(),
	); err != nil {
		return zero, fmt.Errorf("%s: decode strict YAML: %w", file.relative, err)
	}
	if len(documents) != 1 {
		return zero, fmt.Errorf("%s: expected exactly one non-empty YAML document, got %d", file.relative, len(documents))
	}
	return documents[0], nil
}

func (l *loader) loadModelPolicies() error {
	files, err := l.discover("model-policies")
	if err != nil {
		return err
	}
	for _, file := range files {
		document, decodeErr := decodeOne[modelPolicyDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(document.APIVersion, document.Kind, modelPolicyKind, document.Metadata)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.policies[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate ModelPolicy identity %s", file.relative, selector)
		}
		if resolveErr := validateModelPolicySpec(document.Spec); resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		policy := contracts.ResolvedModelPolicy{
			Ref:   contracts.ModelPolicyRef{PolicyID: selector.ID, Version: selector.Version},
			Model: document.Spec.Model, MaxOutputTokens: optionalIntValue(document.Spec.MaxOutputTokens),
			MaxModelCalls: optionalIntValue(document.Spec.MaxModelCalls), MaxToolCalls: optionalIntValue(document.Spec.MaxToolCalls),
			MaxWorkerCalls: optionalIntValue(document.Spec.MaxWorkerCalls), MaxTotalTokens: optionalIntValue(document.Spec.MaxTotalTokens),
			Temperature: cloneFloat(document.Spec.Temperature),
		}
		digest, digestErr := modelPolicyDigest(selector, policy)
		if digestErr != nil {
			return fmt.Errorf("%s: compute ModelPolicy digest: %w", file.relative, digestErr)
		}
		policy.Ref.Digest = digest
		l.policies[selector.String()] = policy
	}
	return nil
}

func optionalIntValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

func (l *loader) loadLLMGatewayConfigs() error {
	files, err := l.discover("llm-gateways")
	if err != nil {
		return err
	}
	for _, file := range files {
		document, decodeErr := decodeOne[llmGatewayConfigDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(
			document.APIVersion, document.Kind, llmGatewayConfigKind, document.Metadata,
		)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.gateways[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate LLMGatewayConfig identity %s", file.relative, selector)
		}
		gateway, resolveErr := resolveLLMGatewayConfig(selector, document.Spec)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		l.gateways[selector.String()] = gateway
	}
	return nil
}

func (l *loader) loadAgentTemplates() error {
	files, err := l.discover("agent-templates")
	if err != nil {
		return err
	}
	for _, file := range files {
		document, decodeErr := decodeOne[agentTemplateDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(document.APIVersion, document.Kind, agentTemplateKind, document.Metadata)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.templates[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate AgentTemplate identity %s", file.relative, selector)
		}
		template, resolveErr := l.resolveAgentTemplate(selector, document.Spec)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		l.templates[selector.String()] = template
	}
	return nil
}

func (l *loader) loadWorkflows() error {
	files, err := l.discover("workflows")
	if err != nil {
		return err
	}
	for _, file := range files {
		document, decodeErr := decodeOne[workflowDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(document.APIVersion, document.Kind, workflowKind, document.Metadata)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.workflows[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate Workflow identity %s", file.relative, selector)
		}
		workflow, resolveErr := l.resolveWorkflow(selector, document.Spec)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		l.workflows[selector.String()] = workflow
	}
	return nil
}

func (l *loader) resolveInstructions(source *instructionsRefSource) (contracts.ResolvedInstructions, error) {
	if source == nil {
		return contracts.ResolvedInstructions{}, fmt.Errorf("instructions are required")
	}
	normalized, err := validateInstructionRef(source.Ref)
	if err != nil {
		return contracts.ResolvedInstructions{}, err
	}
	if existing, ok := l.instructions[normalized]; ok {
		return existing, nil
	}

	candidate := filepath.Join(l.root, filepath.FromSlash(normalized))
	resolved, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return contracts.ResolvedInstructions{}, fmt.Errorf("resolve instruction %q: %w", normalized, err)
	}
	relative, err := filepath.Rel(l.root, resolved)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || filepath.IsAbs(relative) {
		return contracts.ResolvedInstructions{}, fmt.Errorf("instruction ref %q escapes configuration root", normalized)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return contracts.ResolvedInstructions{}, fmt.Errorf("stat instruction %q: %w", normalized, err)
	}
	if !info.Mode().IsRegular() {
		return contracts.ResolvedInstructions{}, fmt.Errorf("instruction ref %q is not a regular file", normalized)
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return contracts.ResolvedInstructions{}, fmt.Errorf("read instruction %q: %w", normalized, err)
	}
	if !utf8.Valid(data) {
		return contracts.ResolvedInstructions{}, fmt.Errorf("instruction ref %q is not strict UTF-8", normalized)
	}
	text := string(data)
	if strings.TrimSpace(text) == "" {
		return contracts.ResolvedInstructions{}, fmt.Errorf("instruction ref %q is empty or whitespace-only", normalized)
	}
	result := contracts.ResolvedInstructions{Ref: normalized, Digest: digestBytes(data), Text: text}
	l.instructions[normalized] = result
	return result, nil
}

func cloneFloat(value *float64) *float64 {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}
