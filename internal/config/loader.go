package config

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

type loader struct {
	roots            []configurationRoot
	strictSymlinks   bool
	descriptors      Descriptors
	instructions     map[string]contracts.ResolvedInstructions
	policies         map[string]contracts.ResolvedModelPolicy
	gateways         map[string]contracts.ResolvedLLMGatewayConfig
	executionConfigs map[string]ResolvedExecutionConfigProfile
	templates        map[string]contracts.ResolvedAgentTemplate
	workflows        map[string]ResolvedWorkflow
	sources          map[string]ConfigurationSource
}

type configurationRoot struct {
	path   string
	source ConfigurationSource
}

type manifestFile struct {
	absolute string
	relative string
	source   ConfigurationSource
}

// Load validates the complete configuration root and publishes one immutable
// snapshot. On any error the returned Snapshot is nil.
func Load(root string, descriptors Descriptors) (*Snapshot, error) {
	return loadConfigurationRoots([]configurationRoot{{path: root, source: ConfigurationSourceOperator}}, descriptors, false)
}

// LoadUnion validates operator and managed roots as one namespace. Duplicate
// manifest identities, logical paths, and any symlink below a fixed subtree
// reject the complete snapshot; neither root has precedence.
func LoadUnion(operatorRoot, managedRoot string, descriptors Descriptors) (*Snapshot, error) {
	return loadConfigurationRoots([]configurationRoot{
		{path: operatorRoot, source: ConfigurationSourceOperator},
		{path: managedRoot, source: ConfigurationSourceManaged},
	}, descriptors, true)
}

func loadConfigurationRoots(
	roots []configurationRoot, descriptors Descriptors, strictSymlinks bool,
) (*Snapshot, error) {
	normalizedDescriptors, err := normalizeDescriptors(descriptors)
	if err != nil {
		return nil, err
	}
	resolvedRoots := make([]configurationRoot, 0, len(roots))
	for _, root := range roots {
		if strictSymlinks {
			absolute, absoluteErr := filepath.Abs(root.path)
			if absoluteErr != nil {
				return nil, fmt.Errorf("%s root: %w", root.source, absoluteErr)
			}
			info, statErr := os.Lstat(absolute)
			if statErr != nil {
				return nil, fmt.Errorf("%s root: %w", root.source, statErr)
			}
			if info.Mode()&os.ModeSymlink != 0 {
				return nil, fmt.Errorf("%s root is a symlink", root.source)
			}
		}
		resolved, resolveErr := resolveRoot(root.path)
		if resolveErr != nil {
			return nil, fmt.Errorf("%s root: %w", root.source, resolveErr)
		}
		resolvedRoots = append(resolvedRoots, configurationRoot{path: resolved, source: root.source})
	}
	for _, root := range resolvedRoots {
		if root.source != ConfigurationSourceOperator {
			continue
		}
		if _, err := agentskills.DiscoverBundled(root.path); err != nil {
			return nil, fmt.Errorf("bundled skills: %w", err)
		}
	}

	current := &loader{
		roots:            resolvedRoots,
		strictSymlinks:   strictSymlinks,
		descriptors:      normalizedDescriptors,
		instructions:     make(map[string]contracts.ResolvedInstructions),
		policies:         make(map[string]contracts.ResolvedModelPolicy),
		gateways:         make(map[string]contracts.ResolvedLLMGatewayConfig),
		executionConfigs: make(map[string]ResolvedExecutionConfigProfile),
		templates:        make(map[string]contracts.ResolvedAgentTemplate),
		workflows:        make(map[string]ResolvedWorkflow),
		sources:          make(map[string]ConfigurationSource),
	}
	if err := current.loadInstructionResources(); err != nil {
		return nil, err
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
		current.executionConfigs, current.instructions, current.sources,
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
	var result []manifestFile
	seen := make(map[string]ConfigurationSource)
	for _, root := range l.roots {
		base := filepath.Join(root.path, subtree)
		if err := validateSubtree(base, subtree, l.strictSymlinks); err != nil {
			return nil, err
		}
		err := filepath.WalkDir(base, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.Type()&os.ModeSymlink != 0 {
				if l.strictSymlinks {
					return fmt.Errorf("configuration path %q is a symlink", path)
				}
				return nil
			}
			if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
				return nil
			}
			if !entry.Type().IsRegular() {
				return fmt.Errorf("configuration manifest %q is not a regular file", path)
			}
			relative, relErr := filepath.Rel(root.path, path)
			if relErr != nil {
				return relErr
			}
			relative = filepath.ToSlash(relative)
			if previous, exists := seen[relative]; exists {
				return fmt.Errorf(
					"duplicate configuration path %q across %s and %s roots",
					relative, previous, root.source,
				)
			}
			seen[relative] = root.source
			result = append(result, manifestFile{
				absolute: path, relative: relative, source: root.source,
			})
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("discover %s manifests in %s root: %w", subtree, root.source, err)
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].relative < result[j].relative })
	return result, nil
}

func validateSubtree(base, subtree string, strictSymlinks bool) error {
	if strictSymlinks {
		info, err := os.Lstat(base)
		if err != nil {
			return fmt.Errorf("configuration subtree %s: %w", subtree, err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("configuration subtree %s is a symlink", subtree)
		}
	}
	info, err := os.Stat(base)
	if err != nil {
		return fmt.Errorf("configuration subtree %s: %w", subtree, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("configuration subtree %s is not a directory", subtree)
	}
	return nil
}

func (l *loader) loadInstructionResources() error {
	seen := make(map[string]ConfigurationSource)
	for _, root := range l.roots {
		base := filepath.Join(root.path, "instructions")
		if err := validateSubtree(base, "instructions", l.strictSymlinks); err != nil {
			return err
		}
		err := filepath.WalkDir(base, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			isSymlink := entry.Type()&os.ModeSymlink != 0
			readPath := path
			if isSymlink {
				if l.strictSymlinks {
					return fmt.Errorf("instruction path %q is a symlink", path)
				}
				resolved, resolveErr := filepath.EvalSymlinks(path)
				if resolveErr != nil {
					return resolveErr
				}
				relativeTarget, relErr := filepath.Rel(root.path, resolved)
				if relErr != nil || relativeTarget == ".." ||
					strings.HasPrefix(relativeTarget, ".."+string(filepath.Separator)) ||
					filepath.IsAbs(relativeTarget) {
					return fmt.Errorf("instruction ref %q escapes configuration root", path)
				}
				info, statErr := os.Stat(resolved)
				if statErr != nil || !info.Mode().IsRegular() {
					return fmt.Errorf("instruction ref %q is not a regular file", path)
				}
				readPath = resolved
			}
			if entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
				return nil
			}
			if !isSymlink && !entry.Type().IsRegular() {
				return fmt.Errorf("instruction path %q is not a regular file", path)
			}
			relative, relErr := filepath.Rel(root.path, path)
			if relErr != nil {
				return relErr
			}
			relative = filepath.ToSlash(relative)
			if _, err := validateInstructionRef(relative); err != nil {
				return err
			}
			if previous, exists := seen[relative]; exists {
				return fmt.Errorf(
					"duplicate instruction path %q across %s and %s roots",
					relative, previous, root.source,
				)
			}
			data, readErr := os.ReadFile(readPath)
			if readErr != nil {
				return fmt.Errorf("read instruction %q: %w", relative, readErr)
			}
			if !utf8.Valid(data) || strings.TrimSpace(string(data)) == "" {
				return fmt.Errorf("instruction ref %q is empty or not strict UTF-8", relative)
			}
			seen[relative] = root.source
			l.instructions[relative] = contracts.ResolvedInstructions{
				Ref: relative, Digest: digestBytes(data), Text: string(data),
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("discover instructions in %s root: %w", root.source, err)
		}
	}
	return nil
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
			Model: document.Spec.Model, ContextWindowTokens: optionalIntValue(document.Spec.ContextWindowTokens), MaxOutputTokens: optionalIntValue(document.Spec.MaxOutputTokens),
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
		l.sources[configurationSourceKey(ConfigurationModelPolicies, selector.String())] = file.source
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
		l.sources[configurationSourceKey(ConfigurationLLMGateways, selector.String())] = file.source
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
		l.sources[configurationSourceKey(ConfigurationAgentTemplates, selector.String())] = file.source
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
	return contracts.ResolvedInstructions{}, fmt.Errorf("unknown instruction ref %q", normalized)
}

func cloneFloat(value *float64) *float64 {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}
