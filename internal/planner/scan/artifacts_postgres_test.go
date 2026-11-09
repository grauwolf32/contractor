package scan

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	contractscan "github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

func TestPostgresScanArtifactReadsUseTheCallerInputBudget(t *testing.T) {
	ctx := t.Context()
	pool := isolatedScanPool(t, ctx)
	_, err := runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), runstore.CreateRunParams{
		LaneClass: resolved.LaneClassTool, RunID: "run-rulepack", OwnerID: "owner", WorkflowName: "scan", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{"name":"scan"}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	scope, err := service.User("owner")
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := NewServiceArtifacts(service)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, media string
		data        []byte
	}{
		{"small-yaml", "application/yaml", []byte("id: fixture\n")},
		{"zip-sized", "application/vnd.contractor.nuclei-rulepack+zip", bytes.Repeat([]byte("x"), planner.MaxScanArtifactBytes+1)},
	} {
		t.Run(test.name, func(t *testing.T) {
			written, err := scope.Write(ctx, contracts.ArtifactRef{Namespace: "rules", Name: test.name}, artifacts.Payload{MediaType: test.media, Data: test.data}, nil)
			if err != nil {
				t.Fatal(err)
			}
			userScope, _ := artifacts.UserScope("owner")
			runScope, _ := artifacts.RunScope("run-rulepack")
			fork, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageInputFork, Source: userScope, SourceRef: written.Ref, Target: runScope, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: test.name}})
			if err != nil {
				t.Fatal(err)
			}
			got, err := adapter.Read(ctx, "run-rulepack", fork.TargetRef, contractscan.MaxNucleiRulepackZIPBytes)
			if err != nil || got.MediaType != test.media || !bytes.Equal(got.Data, test.data) {
				t.Fatalf("permitted input read: %v", err)
			}
			if _, err := adapter.Read(ctx, "run-rulepack", fork.TargetRef, len(test.data)-1); !errors.Is(err, blob.ErrTooLarge) {
				t.Fatalf("narrower caller bound ignored: %v", err)
			}
			if _, err := adapter.Read(ctx, "run-rulepack", fork.TargetRef, contractscan.MaxNucleiRulepackZIPBytes+1); !errors.Is(err, contracts.ErrValidation) {
				t.Fatalf("unsupported read bound accepted: %v", err)
			}
			if _, err := adapter.Create(ctx, "run-rulepack", contracts.ArtifactRef{Namespace: "plans", Name: test.name}, artifacts.Payload{MediaType: "application/json", Data: bytes.Repeat([]byte("x"), planner.MaxScanArtifactBytes+1)}); !errors.Is(err, blob.ErrTooLarge) {
				t.Fatalf("creation ceiling widened: %v", err)
			}
		})
	}
}

func TestPostgresScanMaterialSelectionAdvancesBindingAndRetainsAttempts(t *testing.T) {
	ctx := t.Context()
	pool := isolatedScanPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	_, err := runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool, storage), runstore.CreateRunParams{
		LaneClass: resolved.LaneClassTool, RunID: "run-material-retry", OwnerID: "owner", WorkflowName: "scan", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{"name":"scan"}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := NewServiceArtifacts(artifacts.NewService(storage.Repository(pool)))
	if err != nil {
		t.Fatal(err)
	}
	h := newFactoryHarness(t, 1)
	h.invocation.RunID = "run-material-retry"
	h.factory.artifacts = adapter
	h.invocation.Stage.Result.Artifacts["materials"] = resolved.ArtifactSlot{Required: true, From: &resolved.ArtifactBinding{Namespace: "scan-results", Name: "finding-materials"}}
	p := &execution{factory: h.factory, invocation: h.invocation}
	var first contracts.ArtifactRef
	var firstData []byte
	for _, name := range []string{"attempt-one", "attempt-two"} {
		output, err := adapter.Create(ctx, h.invocation.RunID, contracts.ArtifactRef{Namespace: "scanner", Name: name}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"items":[]}`)})
		if err != nil {
			t.Fatal(err)
		}
		jobs := []planner.ScanJobRecord{{Outputs: map[string]contracts.ArtifactRef{"candidates": output}}}
		result := stageresult.StageContentResult{Artifacts: map[string]contracts.ArtifactRef{}}
		if err := p.selectNormalizedMaterials(ctx, jobs, &result); err != nil {
			t.Fatal(err)
		}
		ref := result.Artifacts["materials"]
		payload, err := adapter.Read(ctx, h.invocation.RunID, ref, planner.MaxScanArtifactBytes)
		if err != nil || ref.Name != "finding-materials" {
			t.Fatalf("downstream binding: %+v %v", ref, err)
		}
		if name == "attempt-one" {
			first, firstData = ref, payload.Data
		} else {
			old, err := adapter.Read(ctx, h.invocation.RunID, first, planner.MaxScanArtifactBytes)
			if err != nil || !bytes.Equal(old.Data, firstData) || first.SameExact(ref) {
				t.Fatal("new attempt changed the previous exact result")
			}
			if _, err := adapter.Replace(ctx, h.invocation.RunID, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name}, payload, *first.Revision); !errors.Is(err, artifacts.ErrArtifactConflict) {
				t.Fatalf("stale attempt overwrote current selection: %v", err)
			}
		}
		if err := p.selectNormalizedMaterials(ctx, jobs, &result); err != nil || !result.Artifacts["materials"].SameExact(ref) {
			t.Fatalf("acknowledgement recovery changed exact selection: %v", err)
		}
	}
}
