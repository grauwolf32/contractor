package config

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestManagerPublishesDurableModelPolicyAndRecoversOnRestart(t *testing.T) {
	operator := copyConfigTree(t)
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, operator, managed, ManagerOptions{})
	before := manager.Counts().ModelPolicies

	request := validPolicyPublication("publish-policy")
	result, err := manager.Publish(t.Context(), request)
	if err != nil {
		t.Fatalf("Publish: %v", err)
	}
	if result.Replayed || result.Resource.Source != ConfigurationSourceManaged {
		t.Fatalf("publication result = %+v", result)
	}
	assertDigest(t, result.Resource.Ref.Digest)
	if got := manager.Counts().ModelPolicies; got != before+1 {
		t.Fatalf("current ModelPolicy count = %d, want baseline+1 (%d)", got, before+1)
	}

	path := filepath.Join(managed, "model-policies", "ui-worker@2.yaml")
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("published file: %v", err)
	}
	if !info.Mode().IsRegular() || info.Mode().Perm()&0o022 != 0 {
		t.Fatalf("published mode = %v", info.Mode())
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil || len(entries) != 1 || entries[0].Name() != filepath.Base(path) {
		t.Fatalf("publication left unexpected files: %v, error %v", entries, err)
	}

	restarted := newTestManager(t, operator, managed, ManagerOptions{})
	recovered, err := restarted.Configuration(ConfigurationModelPolicies, "ui-worker@2")
	if err != nil {
		t.Fatalf("recover published policy: %v", err)
	}
	if recovered.Ref != result.Resource.Ref || recovered.Source != ConfigurationSourceManaged {
		t.Fatalf("recovered = %+v, want %+v", recovered, result.Resource)
	}
}

func TestManagerPublishesNormalizedGateway(t *testing.T) {
	operator := copyConfigTree(t)
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, operator, managed, ManagerOptions{})

	result, err := manager.Publish(t.Context(), PublicationRequest{
		Kind: ConfigurationLLMGateways, Name: "remote", Version: "1",
		IdempotencyKey: "publish-gateway", ActorID: "user-1",
		LLMGateway: &LLMGatewayPublication{
			Protocol: contracts.OpenAICompatibleProtocol,
			URL:      "https://gateway.example.test/v1/",
			CredentialManager: &CredentialManagerPublication{
				Implementation: contracts.LiteLLMVirtualKeysManager,
				ManagementURL:  "https://gateway.example.test/",
			},
		},
	})
	if err != nil {
		t.Fatalf("Publish: %v", err)
	}
	body := result.Resource.Body.(map[string]any)
	managerBody := body["credentialManager"].(map[string]any)
	if managerBody["managementUrl"] != "https://gateway.example.test" {
		t.Fatalf("normalized credential manager = %#v", managerBody)
	}
	restarted := newTestManager(t, operator, managed, ManagerOptions{})
	gateway, err := restarted.Snapshot().LLMGateway("remote@1")
	if err != nil || gateway.Ref.Digest != result.Resource.Ref.Digest {
		t.Fatalf("restarted Gateway = %+v, error %v", gateway, err)
	}
}

func TestManagerPublicationIdempotencyAndConflicts(t *testing.T) {
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, copyConfigTree(t), managed, ManagerOptions{})
	request := validPolicyPublication("stable-key")
	created, err := manager.Publish(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := manager.Publish(t.Context(), request)
	if err != nil || !replayed.Replayed || replayed.Resource.Ref != created.Resource.Ref {
		t.Fatalf("same-key replay = %+v, error %v", replayed, err)
	}

	different := request
	different.Name = "another-policy"
	if _, err := manager.Publish(t.Context(), different); !errors.Is(err, ErrPublicationConflict) {
		t.Fatalf("same key with another request error = %v", err)
	}

	differentKey := request
	differentKey.IdempotencyKey = "another-key"
	replayed, err = manager.Publish(t.Context(), differentKey)
	if err != nil || !replayed.Replayed {
		t.Fatalf("same durable identity/body replay = %+v, error %v", replayed, err)
	}

	differentBody := request
	differentBody.IdempotencyKey = "different-body"
	differentBody.ModelPolicy = &ModelPolicyPublication{Model: "another-model"}
	if _, err := manager.Publish(t.Context(), differentBody); !errors.Is(err, ErrPublicationConflict) {
		t.Fatalf("immutable identity overwrite error = %v", err)
	}

	if err := os.Remove(filepath.Join(managed, "model-policies", "ui-worker@2.yaml")); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Publish(t.Context(), request); err == nil {
		t.Fatal("same-key replay succeeded after the durable manifest was removed")
	}
}

func TestManagerCrashAfterPublicationIsRecovered(t *testing.T) {
	operator := copyConfigTree(t)
	managed := filepath.Join(t.TempDir(), "managed")
	crash := errors.New("simulated crash")
	manager := newTestManager(t, operator, managed, ManagerOptions{
		AfterDurablePublish: func(ConfigurationResource) error { return crash },
	})
	request := validPolicyPublication("crash-window")
	if _, err := manager.Publish(t.Context(), request); !errors.Is(err, crash) {
		t.Fatalf("Publish error = %v", err)
	}
	if _, err := manager.Configuration(ConfigurationModelPolicies, "ui-worker@2"); !errors.Is(err, ErrConfigurationNotFound) {
		t.Fatalf("old in-memory snapshot changed: %v", err)
	}

	// Reproduce the directory state of a crash between publishing the final name
	// and removing the temporary name. Startup must load the manifest only once.
	directory := filepath.Join(managed, "model-policies")
	if err := os.Link(filepath.Join(directory, "ui-worker@2.yaml"), filepath.Join(directory, ".contractor-publish-interrupted.tmp")); err != nil {
		t.Fatal(err)
	}
	restarted := newTestManager(t, operator, managed, ManagerOptions{})
	if _, err := restarted.Configuration(ConfigurationModelPolicies, "ui-worker@2"); err != nil {
		t.Fatalf("restart did not recover durable version: %v", err)
	}
}

func TestManagerDurablePublicationPreservesExistingEntries(t *testing.T) {
	for _, kind := range []string{"file", "directory", "symlink", "dangling-symlink"} {
		t.Run(kind, func(t *testing.T) {
			managed := filepath.Join(t.TempDir(), "managed")
			manager := newTestManager(t, copyConfigTree(t), managed, ManagerOptions{})
			candidate, err := preparePublication(validPolicyPublication("existing-entry"))
			if err != nil {
				t.Fatal(err)
			}
			destination := filepath.Join(managed, "model-policies", "ui-worker@2.yaml")
			target := filepath.Join(t.TempDir(), "target")
			original := []byte("original content\n")
			switch kind {
			case "file":
				writeFile(t, destination, original)
			case "directory":
				if err := os.Mkdir(destination, 0o750); err != nil {
					t.Fatal(err)
				}
			case "symlink", "dangling-symlink":
				if kind == "symlink" {
					writeFile(t, target, original)
				}
				if err := os.Symlink(target, destination); err != nil {
					t.Fatal(err)
				}
			}
			before, err := os.Lstat(destination)
			if err != nil {
				t.Fatal(err)
			}
			// Bypass Publish's snapshot precheck to exercise a destination created
			// by another writer just before the filesystem publication.
			if err := manager.writeDurableManifest(candidate); !errors.Is(err, ErrPublicationConflict) {
				t.Fatalf("existing %s: expected publication conflict, got %v", kind, err)
			}
			after, err := os.Lstat(destination)
			if err != nil || !os.SameFile(before, after) || before.Mode() != after.Mode() {
				t.Fatalf("existing %s was replaced: %v", kind, err)
			}
			if kind == "file" || kind == "symlink" {
				if got := readFile(t, destination); !bytes.Equal(got, original) {
					t.Fatalf("existing content changed: %q", got)
				}
			}
			if kind == "dangling-symlink" {
				if _, err := os.Lstat(target); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("dangling symlink target was created: %v", err)
				}
			}
			entries, err := os.ReadDir(filepath.Dir(destination))
			if err != nil || len(entries) != 1 {
				t.Fatalf("conflict left unexpected files: %v, error %v", entries, err)
			}
		})
	}
}

func TestManagerConcurrentDurablePublicationsHaveOneWinner(t *testing.T) {
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, copyConfigTree(t), managed, ManagerOptions{})
	const writers = 8
	candidates := make([]publicationCandidate, writers)
	for index := range candidates {
		request := validPolicyPublication(fmt.Sprintf("writer-%d", index))
		request.ModelPolicy.Model = fmt.Sprintf("model-%d", index)
		candidate, err := preparePublication(request)
		if err != nil {
			t.Fatal(err)
		}
		candidates[index] = candidate
	}
	type publicationOutcome struct {
		manifest []byte
		err      error
	}
	start := make(chan struct{})
	outcomes := make(chan publicationOutcome, writers)
	for _, candidate := range candidates {
		go func() {
			<-start
			outcomes <- publicationOutcome{candidate.canonicalYAML, manager.writeDurableManifest(candidate)}
		}()
	}
	close(start)
	var winner []byte
	successes := 0
	for range writers {
		outcome := <-outcomes
		if outcome.err == nil {
			successes++
			winner = outcome.manifest
		} else if !errors.Is(outcome.err, ErrPublicationConflict) {
			t.Errorf("unexpected publication error: %v", outcome.err)
		}
	}
	if successes != 1 {
		t.Fatalf("successful publications = %d, want 1", successes)
	}
	destination := filepath.Join(managed, "model-policies", "ui-worker@2.yaml")
	if got := readFile(t, destination); !bytes.Equal(got, winner) {
		t.Fatal("published manifest differs from the successful writer's complete content")
	}
	entries, err := os.ReadDir(filepath.Dir(destination))
	if err != nil || len(entries) != 1 {
		t.Fatalf("concurrent publications left unexpected files: %v, error %v", entries, err)
	}
}

func TestManagerPublicationIsAtomicForConcurrentReaders(t *testing.T) {
	manager := newTestManager(t, copyConfigTree(t), filepath.Join(t.TempDir(), "managed"), ManagerOptions{})
	before := manager.Counts().ModelPolicies
	start := make(chan struct{})
	var wait sync.WaitGroup
	for range 16 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			for range 100 {
				snapshot := manager.Snapshot()
				count := snapshot.Counts().ModelPolicies
				if count != before && count != before+1 {
					t.Errorf("reader observed partial count %d", count)
					return
				}
				_, err := snapshot.ModelPolicy("ui-worker@2")
				if count == before && err == nil || count == before+1 && err != nil {
					t.Errorf("reader observed inconsistent snapshot count=%d err=%v", count, err)
					return
				}
			}
		}()
	}
	close(start)
	if _, err := manager.Publish(t.Context(), validPolicyPublication("concurrent")); err != nil {
		t.Fatal(err)
	}
	wait.Wait()
}

func TestManagerRejectsInvalidPublicationBeforeFilesystemChange(t *testing.T) {
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, copyConfigTree(t), managed, ManagerOptions{})
	tests := []PublicationRequest{
		{Kind: ConfigurationAgentTemplates, Name: "worker", Version: "2", IdempotencyKey: "readonly"},
		{Kind: ConfigurationModelPolicies, Name: "../escape", Version: "1", IdempotencyKey: "escape", ModelPolicy: &ModelPolicyPublication{Model: "m"}},
		{Kind: ConfigurationModelPolicies, Name: "zero", Version: "1", IdempotencyKey: "zero", ModelPolicy: &ModelPolicyPublication{Model: "m", MaxModelCalls: intPointer(0)}},
		{Kind: ConfigurationModelPolicies, Name: "no-output-reserve", Version: "1", IdempotencyKey: "no-output-reserve", ModelPolicy: &ModelPolicyPublication{Model: "m", ContextWindowTokens: intPointer(4096), MaxOutputTokens: intPointer(4096)}},
		{Kind: ConfigurationModelPolicies, Name: "hot", Version: "1", IdempotencyKey: "hot", ModelPolicy: &ModelPolicyPublication{Model: "m", Temperature: floatPointer(-0.1)}},
		{Kind: ConfigurationModelPolicies, Name: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Version: "1", IdempotencyKey: "long-name", ModelPolicy: &ModelPolicyPublication{Model: "m"}},
	}
	for _, request := range tests {
		if _, err := manager.Publish(t.Context(), request); err == nil {
			t.Errorf("invalid publication %+v succeeded", request)
		}
	}
	entries, err := filepath.Glob(filepath.Join(managed, "model-policies", "*.yaml"))
	if err != nil || len(entries) != 0 {
		t.Fatalf("invalid requests created files %v, error %v", entries, err)
	}
}

func TestManagerAuditIsBestEffortAndSecretFree(t *testing.T) {
	recorder := &recordingPublicationAudit{err: errors.New("database unavailable")}
	manager := newTestManager(t, copyConfigTree(t), filepath.Join(t.TempDir(), "managed"), ManagerOptions{
		Audit: recorder,
		Now:   func() time.Time { return time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC) },
	})
	request := validPolicyPublication("private-idempotency-key")
	request.ActorID = "user-1"
	if _, err := manager.Publish(t.Context(), request); err != nil {
		t.Fatalf("audit failure changed publication result: %v", err)
	}
	if len(recorder.values) != 1 {
		t.Fatalf("audit calls = %d", len(recorder.values))
	}
	audit := recorder.values[0]
	if audit.IdempotencyKeyDigest == request.IdempotencyKey || audit.ActorID != "user-1" || audit.PublishedAt.IsZero() {
		t.Fatalf("unsafe or incomplete audit = %+v", audit)
	}
}

func TestLoadUnionRejectsDuplicatesAndSymlinks(t *testing.T) {
	operator := copyConfigTree(t)
	managed := filepath.Join(t.TempDir(), "managed")
	if _, err := requireStrictRoot(managed, true); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(managed, "model-policies", "duplicate.yaml"), readFile(t, filepath.Join(operator, "model-policies", "worker.yaml")))
	if _, err := LoadUnion(operator, managed, MVPDescriptors()); err == nil {
		t.Fatal("duplicate identity across roots was accepted")
	}
	if err := os.Remove(filepath.Join(managed, "model-policies", "duplicate.yaml")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(operator, "model-policies", "worker.yaml"), filepath.Join(managed, "model-policies", "linked.yaml")); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadUnion(operator, managed, MVPDescriptors()); err == nil {
		t.Fatal("symlink in union root was accepted")
	}
}

func TestManagerRejectsSymlinkRootAndSubtree(t *testing.T) {
	operator := copyConfigTree(t)
	parent := t.TempDir()
	realManaged := filepath.Join(parent, "real")
	if _, err := requireStrictRoot(realManaged, true); err != nil {
		t.Fatal(err)
	}
	linkedManaged := filepath.Join(parent, "linked")
	if err := os.Symlink(realManaged, linkedManaged); err != nil {
		t.Fatal(err)
	}
	if _, err := NewManager(ManagerOptions{OperatorRoot: operator, ManagedRoot: linkedManaged, Descriptors: MVPDescriptors()}); err == nil {
		t.Fatal("symlink managed root was accepted")
	}

	other := filepath.Join(t.TempDir(), "managed")
	if err := os.MkdirAll(other, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, subtree := range configurationSubtrees {
		if subtree == "model-policies" {
			if err := os.Symlink(filepath.Join(operator, subtree), filepath.Join(other, subtree)); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := os.Mkdir(filepath.Join(other, subtree), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := NewManager(ManagerOptions{OperatorRoot: operator, ManagedRoot: other, Descriptors: MVPDescriptors()}); err == nil {
		t.Fatal("symlink managed subtree was accepted")
	}
}

func TestManagerFailsClosedOnExternalFilesystemChange(t *testing.T) {
	operator := copyConfigTree(t)
	managed := filepath.Join(t.TempDir(), "managed")
	manager := newTestManager(t, operator, managed, ManagerOptions{})
	writeFile(t, filepath.Join(managed, "model-policies", "external@1.yaml"), []byte(`apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: external, version: "1"}
spec: {model: external-model}
`))
	request := validPolicyPublication("after-external-change")
	if _, err := manager.Publish(t.Context(), request); err == nil {
		t.Fatal("publication accepted uncoordinated filesystem drift")
	}
	if _, err := manager.Configuration(ConfigurationModelPolicies, "external@1"); !errors.Is(err, ErrConfigurationNotFound) {
		t.Fatalf("external change leaked into current snapshot: %v", err)
	}
}

type recordingPublicationAudit struct {
	values []PublicationAudit
	err    error
}

func (r *recordingPublicationAudit) RecordConfigurationPublication(_ context.Context, value PublicationAudit) error {
	r.values = append(r.values, value)
	return r.err
}

func newTestManager(t *testing.T, operator, managed string, overrides ManagerOptions) *Manager {
	t.Helper()
	overrides.OperatorRoot = operator
	overrides.ManagedRoot = managed
	overrides.Descriptors = MVPDescriptors()
	manager, err := NewManager(overrides)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	return manager
}

func validPolicyPublication(key string) PublicationRequest {
	return PublicationRequest{
		Kind: ConfigurationModelPolicies, Name: "ui-worker", Version: "2",
		IdempotencyKey: key,
		ModelPolicy: &ModelPolicyPublication{
			Model: "qwen/new-model", ContextWindowTokens: intPointer(131_072),
			MaxOutputTokens: intPointer(4096),
			MaxModelCalls:   intPointer(8), MaxToolCalls: intPointer(16),
			MaxTotalTokens: intPointer(32768), Temperature: floatPointer(0.2),
		},
	}
}

func intPointer(value int) *int           { return &value }
func floatPointer(value float64) *float64 { return &value }
