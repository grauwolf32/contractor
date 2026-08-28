package config

import (
	"path/filepath"
	"testing"
)

func TestSemanticDigestsIgnoreYAMLPresentationAndToolOrder(t *testing.T) {
	t.Parallel()

	baselineRoot := copyConfigTree(t)
	variantRoot := copyConfigTree(t)
	writePresentationVariant(t, variantRoot)

	baseline := mustLoad(t, baselineRoot, MVPDescriptors())
	variant := mustLoad(t, variantRoot, MVPDescriptors())
	baselinePolicy, _ := baseline.ModelPolicy("worker@1")
	variantPolicy, _ := variant.ModelPolicy("worker@1")
	if baselinePolicy.Ref.Digest != variantPolicy.Ref.Digest {
		t.Fatalf("presentation changed ModelPolicy digest: %s != %s", baselinePolicy.Ref.Digest, variantPolicy.Ref.Digest)
	}
	baselineTemplate, _ := baseline.AgentTemplate("artifact_builder@1")
	variantTemplate, _ := variant.AgentTemplate("artifact_builder@1")
	if baselineTemplate.Ref.Digest != variantTemplate.Ref.Digest {
		t.Fatalf("presentation changed AgentTemplate digest: %s != %s", baselineTemplate.Ref.Digest, variantTemplate.Ref.Digest)
	}
}

func TestSemanticAndInstructionChangesAlterDigests(t *testing.T) {
	t.Parallel()

	baselineRoot := copyConfigTree(t)
	writePresentationVariant(t, baselineRoot)
	semanticRoot := copyConfigTree(t)
	writePresentationVariant(t, semanticRoot)
	replaceFile(t, filepath.Join(semanticRoot, "model-policies/worker.yaml"), "maxOutputTokens: 4096", "maxOutputTokens: 4097")
	instructionRoot := copyConfigTree(t)
	writePresentationVariant(t, instructionRoot)
	appendFile(t, filepath.Join(instructionRoot, "instructions/artifact-builder.md"), "\n")

	baseline := mustLoad(t, baselineRoot, MVPDescriptors())
	semantic := mustLoad(t, semanticRoot, MVPDescriptors())
	instruction := mustLoad(t, instructionRoot, MVPDescriptors())
	baselinePolicy, _ := baseline.ModelPolicy("worker@1")
	semanticPolicy, _ := semantic.ModelPolicy("worker@1")
	instructionPolicy, _ := instruction.ModelPolicy("worker@1")
	if baselinePolicy.Ref.Digest == semanticPolicy.Ref.Digest {
		t.Fatal("semantic ModelPolicy change did not alter its digest")
	}
	if baselinePolicy.Ref.Digest != instructionPolicy.Ref.Digest {
		t.Fatal("instruction change unexpectedly altered ModelPolicy digest")
	}

	baselineTemplate, _ := baseline.AgentTemplate("artifact_builder@1")
	semanticTemplate, _ := semantic.AgentTemplate("artifact_builder@1")
	instructionTemplate, _ := instruction.AgentTemplate("artifact_builder@1")
	if baselineTemplate.Ref.Digest == semanticTemplate.Ref.Digest {
		t.Fatal("resolved ModelPolicy change did not alter AgentTemplate digest")
	}
	if baselineTemplate.Ref.Digest == instructionTemplate.Ref.Digest {
		t.Fatal("instruction byte change did not alter AgentTemplate digest")
	}
	if baselineTemplate.Instructions.Digest == instructionTemplate.Instructions.Digest {
		t.Fatal("instruction byte change did not alter instruction digest")
	}
}

func TestInstructionDigestUsesExactBytes(t *testing.T) {
	t.Parallel()

	first := []byte("same text\n")
	second := []byte("same text\r\n")
	if digestBytes(first) == digestBytes(second) {
		t.Fatal("line-ending normalization affected exact-byte digest")
	}
}

func writePresentationVariant(t *testing.T, root string) {
	t.Helper()
	writeFile(t, filepath.Join(root, "model-policies/worker.yaml"), []byte(`# key order and scalar spelling are intentionally different
kind: ModelPolicy
apiVersion: contractor/v1alpha1
spec:
  temperature: 0.10
  maxOutputTokens: 4096
  model: worker-model
metadata: {version: "1", name: worker}
`))
	writeFile(t, filepath.Join(root, "agent-templates/artifact_builder.yaml"), []byte(`# Toolset/tool ordering is semantically unordered.
kind: AgentTemplate
metadata: {version: "1", name: artifact_builder}
apiVersion: contractor/v1alpha1
spec:
  sandboxProfile: local-workdir@1
  toolsets:
    - tools: [write_artifact, read_artifact, list_artifacts]
      ref: run-artifacts@1
  modelPolicy: worker@1
  instructions: {ref: instructions/artifact-builder.md}
  runtime: adk@1
  description: "Reads a declared input and writes the requested result"
`))
}
