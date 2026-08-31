package runtimeconfig

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestMergeSameLayerIsOrderIndependent(t *testing.T) {
	t.Parallel()
	one := LayerEntry{Label: "a", Ref: testRef("a", "1"), Spec: Spec{Worker: WorkerPatch{
		LLMGateway: LLMGatewayPatch{Present: true, Gateway: Field[contracts.LLMGatewayConfigRef]{Present: true, Value: contracts.LLMGatewayConfigRef{
			GatewayID: "local", Version: "1", Digest: "sha256:" + strings.Repeat("1", 64),
		}}},
	}}}
	two := LayerEntry{Label: "b", Ref: testRef("b", "2"), Spec: Spec{Worker: WorkerPatch{
		LLMGateway: LLMGatewayPatch{Present: true, Credential: Field[string]{Present: true, Clear: true}},
	}}}
	left, err := MergeSameLayer([]LayerEntry{one, two})
	if err != nil {
		t.Fatal(err)
	}
	right, err := MergeSameLayer([]LayerEntry{two, one})
	if err != nil {
		t.Fatal(err)
	}
	if !specEqual(left, right) || !left.Worker.LLMGateway.Gateway.Present || !left.Worker.LLMGateway.Credential.Clear {
		t.Fatalf("independent merge mismatch: %+v / %+v", left, right)
	}
}

func TestMergeSameLayerReturnsDeterministicSafeConflict(t *testing.T) {
	t.Parallel()
	makeEntry := func(name, endpoint string) LayerEntry {
		return LayerEntry{Label: name, Ref: testRef(name, "1"), Spec: Spec{Worker: WorkerPatch{Telemetry: AtomicPatch[TelemetryConfig]{
			Present: true, Value: TelemetryConfig{Adapter: "otlp-http@1", Endpoint: endpoint, FlushTimeoutSeconds: 3},
		}}}}
	}
	one, two := makeEntry("one", "https://one.example/v1/traces"), makeEntry("two", "https://two.example/v1/traces")
	for _, entries := range [][]LayerEntry{{one, two}, {two, one}} {
		_, err := MergeSameLayer(entries)
		var conflict *MergeConflictError
		if !errors.As(err, &conflict) || conflict.Path != "worker.telemetry" || len(conflict.Refs) != 2 || conflict.Refs[0].Name != "one" || conflict.Refs[1].Name != "two" {
			t.Fatalf("merge conflict = %#v (%v)", conflict, err)
		}
		if strings.Contains(err.Error(), "https://") {
			t.Fatalf("merge error exposed configuration content: %v", err)
		}
	}
}

func testRef(name, digit string) Ref {
	return Ref{Name: name, Version: "1", Digest: "sha256:" + strings.Repeat(digit, 64)}
}
