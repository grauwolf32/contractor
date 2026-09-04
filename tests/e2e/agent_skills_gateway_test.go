//go:build e2e

package e2e

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

type agentSkillMVPFixture struct {
	SchemaVersion      string   `json:"schemaVersion"`
	SkillName          string   `json:"skillName"`
	ResourcePath       string   `json:"resourcePath"`
	PackageACanary     string   `json:"packageACanary"`
	PackageBCanary     string   `json:"packageBCanary"`
	NativeToolSequence []string `json:"nativeToolSequence"`
}

func agentSkillGatewayStages(fixture agentSkillMVPFixture) []domainGatewayStage {
	likeC4BuilderTools := []string{
		"append_likec4", "list_source_files", "load_likec4", "open_source_archive",
		"read_likec4", "read_source", "read_text_artifact", "replace_likec4",
		"search_source", "validate_likec4", "write_likec4",
	}
	likeC4ValidatorTools := []string{
		"append_likec4", "load_likec4", "open_source_archive", "read_likec4", "read_source",
		"read_text_artifact", "replace_likec4", "search_source", "validate_likec4",
		"write_likec4", "write_text_artifact",
	}

	return []domainGatewayStage{
		artifactCopyGatewayStage("blocker/copy"),
		failedAgentSkillGatewayStage("old/likec4_build_attempt_1", likeC4BuilderTools, fixture, fixture.PackageACanary),
		withAgentSkillGateway(
			renameGatewayStage(likeC4BuildGatewayStage(likeC4BuilderTools), "old/likec4_build_attempt_2"),
			fixture, fixture.PackageACanary,
		),
		withAgentSkillGateway(
			renameGatewayStage(likeC4ValidateGatewayStage(likeC4ValidatorTools), "old/likec4_validate"),
			fixture, fixture.PackageACanary,
		),
		withAgentSkillGateway(
			renameGatewayStage(likeC4BuildGatewayStage(likeC4BuilderTools), "new/likec4_build"),
			fixture, fixture.PackageBCanary,
		),
		withAgentSkillGateway(
			renameGatewayStage(likeC4ValidateGatewayStage(likeC4ValidatorTools), "new/likec4_validate"),
			fixture, fixture.PackageBCanary,
		),
		artifactCopyGatewayStage("reuse/copy"),
	}
}

func renameGatewayStage(stage domainGatewayStage, name string) domainGatewayStage {
	stage.name = name
	return stage
}

func artifactCopyGatewayStage(name string) domainGatewayStage {
	return domainGatewayStage{
		name: name, tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
		steps: []domainGatewayStep{
			toolGatewayStep("read_artifact", fixedArguments(map[string]any{
				"namespace": "inputs", "name": "source", "revision": nil,
			})),
			toolGatewayStep("write_artifact", func(request map[string]any) (map[string]any, error) {
				data, ok := lastStringValue(request, "dataBase64")
				if !ok || data == "" {
					return nil, errors.New("copy input was not returned by read_artifact")
				}
				return map[string]any{
					"namespace": "builder", "name": "copied", "media_type": "text/plain",
					"data_base64": data, "expected_revision": nil,
				}, nil
			}),
			finalGatewayStep("Source artifact copied", map[string]domainArtifactBinding{
				"copied": {namespace: "builder", name: "copied"},
			}),
		},
	}
}

func withAgentSkillGateway(
	stage domainGatewayStage,
	fixture agentSkillMVPFixture,
	canary string,
) domainGatewayStage {
	stage.tools = append(append([]string(nil), stage.tools...), fixture.NativeToolSequence...)
	sort.Strings(stage.tools)
	prefix := agentSkillDisclosureSteps(fixture)
	if len(stage.steps) == 0 {
		panic("Agent Skill gateway Stage has no domain steps")
	}
	first := stage.steps[0]
	first.validate = composeGatewayValidation(requireToolOutput(canary), first.validate)
	stage.steps = append(prefix, append([]domainGatewayStep{first}, stage.steps[1:]...)...)
	return stage
}

func failedAgentSkillGatewayStage(
	name string,
	tools []string,
	fixture agentSkillMVPFixture,
	canary string,
) domainGatewayStage {
	tools = append(append([]string(nil), tools...), fixture.NativeToolSequence...)
	sort.Strings(tools)
	steps := agentSkillDisclosureSteps(fixture)
	steps = append(steps, domainGatewayStep{
		validate: requireToolOutput(canary), modelFail: true,
	})
	return domainGatewayStage{name: name, tools: tools, steps: steps}
}

func agentSkillDisclosureSteps(fixture agentSkillMVPFixture) []domainGatewayStep {
	return []domainGatewayStep{
		toolGatewayStep("list_skills", fixedArguments(map[string]any{})),
		{
			tool: "load_skill",
			validate: composeGatewayValidation(
				requireToolOutput("<available_skills>"),
				requireToolOutput(fixture.SkillName),
			),
			arguments: fixedArguments(map[string]any{"skill_name": fixture.SkillName}),
		},
		{
			tool: "load_skill_resource", validate: requireToolOutput("# LikeC4 DSL Skill"),
			arguments: fixedArguments(map[string]any{
				"skill_name": fixture.SkillName, "file_path": fixture.ResourcePath,
			}),
		},
	}
}

func requireToolOutput(fragment string) func(map[string]any) error {
	return func(request map[string]any) error {
		messages, ok := request["messages"].([]any)
		if !ok || len(messages) == 0 {
			return errors.New("native Skill tool response is absent")
		}
		// ADK/LiteLLM may place the function response before a synthesized user
		// message or JSON-encode its content once more, so message order and
		// whitespace escapes are not stable transport contracts. The selected
		// fragment cannot occur before the scripted native call; finding it
		// anywhere in the next model input proves disclosure.
		if !containsStringFragment(messages, fragment) {
			return errors.New("native Skill tool response did not match the fixture")
		}
		return nil
	}
}

func containsStringFragment(value any, fragment string) bool {
	switch typed := value.(type) {
	case string:
		return strings.Contains(typed, fragment)
	case []any:
		for _, item := range typed {
			if containsStringFragment(item, fragment) {
				return true
			}
		}
	case map[string]any:
		for _, item := range typed {
			if containsStringFragment(item, fragment) {
				return true
			}
		}
	}
	return false
}

func composeGatewayValidation(
	first, second func(map[string]any) error,
) func(map[string]any) error {
	return func(request map[string]any) error {
		for index, validate := range []func(map[string]any) error{first, second} {
			if validate != nil {
				if err := validate(request); err != nil {
					return fmt.Errorf("validation %d: %w", index+1, err)
				}
			}
		}
		return nil
	}
}
