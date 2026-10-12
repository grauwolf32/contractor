package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
)

// TestManagedPublicationNeverRebindsShippedIdentity pins the runtime half of
// CAT-R44: managed publication cannot change an identity that the operator
// root ships, with different or with identical content, and leaves both roots
// and the served configuration untouched.
func TestManagedPublicationNeverRebindsShippedIdentity(t *testing.T) {
	operator := copyCoreFixture(t)
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, operator, managed, ManagerOptions{})
	shippedPath := filepath.Join(operator, "model-policies", "test-worker.yaml")
	shippedBytes := readFile(t, shippedPath)
	shipped, err := manager.Configuration(ConfigurationModelPolicies, "test-worker@1")
	if err != nil || shipped.Source != ConfigurationSourceOperator {
		t.Fatalf("shipped policy = %+v, %v", shipped, err)
	}
	for _, test := range []struct {
		name        string
		temperature float64
	}{{"changed", 0.6}, {"identical", 0.1}} {
		t.Run(test.name, func(t *testing.T) {
			request := PublicationRequest{
				Kind: ConfigurationModelPolicies, Name: "test-worker", Version: "1",
				IdempotencyKey: "rebind-" + test.name,
				ModelPolicy: &ModelPolicyPublication{
					Model: "worker-model", MaxOutputTokens: intPointer(4096),
					MaxModelCalls: intPointer(8), MaxToolCalls: intPointer(16),
					MaxTotalTokens: intPointer(32768), Temperature: floatPointer(test.temperature),
				},
			}
			candidate, err := preparePublication(request)
			if err != nil || (candidate.resource.Ref.Digest == shipped.Ref.Digest) != (test.name == "identical") {
				t.Fatalf("candidate digest %s for shipped %s: %v", candidate.resource.Ref.Digest, shipped.Ref.Digest, err)
			}
			_, err = manager.Publish(t.Context(), request)
			if !errors.Is(err, ErrPublicationConflict) {
				t.Fatalf("publication over a shipped identity = %v", err)
			}
			current, err := manager.Configuration(ConfigurationModelPolicies, "test-worker@1")
			if err != nil || current.Ref != shipped.Ref || current.Source != ConfigurationSourceOperator {
				t.Fatalf("served policy after the conflict = %+v, %v", current, err)
			}
		})
	}
	if !bytes.Equal(readFile(t, shippedPath), shippedBytes) {
		t.Fatal("publication changed the shipped manifest")
	}
	if entries, err := os.ReadDir(filepath.Join(managed, "model-policies")); err != nil || len(entries) != 0 {
		t.Fatalf("managed root after the conflicts = %v, %v", entries, err)
	}
}

// TestShippedCatalogInPlaceChangeIsVisibleInRunSnapshots pins CAT-R74 and what
// still detects such a change. Until the first release an in-place edit of a
// shipped manifest loads under the same identity at the next start. The
// digest-bearing refs of a new Run snapshot change with the content they
// identify, and the Eval-style pinned selection digest rejects the change even
// when only the Workflow, which has no digest of its own, changed. Once the
// shipped catalog gets its digest lock, the restart below must fail instead.
func TestShippedCatalogInPlaceChangeIsVisibleInRunSnapshots(t *testing.T) {
	for _, test := range []struct {
		name, file, from, to string
		dependencyChanged    bool
	}{
		{
			name: "model policy", file: "model-policies/test-worker.yaml",
			from: "temperature: 0.1", to: "temperature: 0.6", dependencyChanged: true,
		},
		{
			name: "workflow", file: "workflows/artifact_copy.yaml",
			from: "objective: Copy the source artifact into the declared result artifact",
			to:   "objective: Copy the source artifact into the result artifact",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			operator := copyCoreFixture(t)
			managed := filepath.Join(t.TempDir(), "managed")
			before := resolveShippedRun(t, newTestManager(t, operator, managed, ManagerOptions{}))
			pin, err := contentdigest.JSON(before)
			if err != nil {
				t.Fatal(err)
			}

			replaceFile(t, filepath.Join(operator, test.file), test.from, test.to)
			after := resolveShippedRun(t, newTestManager(t, operator, managed, ManagerOptions{}))

			if after.Ref != before.Ref {
				t.Fatalf("Workflow identity changed: %+v -> %+v", before.Ref, after.Ref)
			}
			oldTemplate := before.Stages["copy"].Agents["builder"].Template
			newTemplate := after.Stages["copy"].Agents["builder"].Template
			oldPolicy := before.Stages["copy"].ExecutionConfig.Agents["builder"].ModelPolicy.Ref
			newPolicy := after.Stages["copy"].ExecutionConfig.Agents["builder"].ModelPolicy.Ref
			if newTemplate.Ref.TemplateID != oldTemplate.Ref.TemplateID ||
				newTemplate.Ref.Version != oldTemplate.Ref.Version ||
				newPolicy.PolicyID != oldPolicy.PolicyID || newPolicy.Version != oldPolicy.Version {
				t.Fatalf("dependency identities changed: %+v/%+v -> %+v/%+v",
					oldTemplate.Ref, oldPolicy, newTemplate.Ref, newPolicy)
			}
			if (newTemplate.Ref.Digest != oldTemplate.Ref.Digest) != test.dependencyChanged ||
				(newPolicy.Digest != oldPolicy.Digest) != test.dependencyChanged {
				t.Fatalf("dependency digests: template %s -> %s, policy %s -> %s",
					oldTemplate.Ref.Digest, newTemplate.Ref.Digest, oldPolicy.Digest, newPolicy.Digest)
			}
			if err := contentdigest.CheckPinnedSelection(pin, after); !errors.Is(err, contentdigest.ErrPinnedSelectionChanged) {
				t.Fatalf("pinned selection accepted changed shipped content: %v", err)
			}
			beforeSnapshot, err := json.Marshal(before)
			if err != nil {
				t.Fatal(err)
			}
			afterSnapshot, err := json.Marshal(after)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Equal(beforeSnapshot, afterSnapshot) {
				t.Fatal("Run snapshots of different content are equal")
			}
		})
	}
}

func resolveShippedRun(t *testing.T, manager *Manager) ResolvedWorkflow {
	t.Helper()
	workflow, err := manager.ResolveRunWorkflow(
		t.Context(), "artifact-copy@1", ExecutionConfigPatch{},
		developmentCredentialLookup(t, manager.Snapshot()),
	)
	if err != nil {
		t.Fatal(err)
	}
	return workflow
}
