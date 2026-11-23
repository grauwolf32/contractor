package memory

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/memorycontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedFindingMemorySources(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "finding-memory-sources.json")
	var fixture struct {
		Limits struct {
			SnapshotBytes    int `json:"snapshotBytes"`
			SnapshotMemories int `json:"snapshotMemories"`
			FactCalls        int `json:"factCalls"`
		} `json:"limits"`
		Cases []struct {
			Name           string `json:"name"`
			SnapshotJSON   string `json:"snapshotJson"`
			SnapshotValid  bool   `json:"snapshotValid"`
			MemoryName     string `json:"memoryName"`
			SelectionValid bool   `json:"selectionValid"`
		} `json:"cases"`
	}
	if err := strictjson.Decode(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Limits.SnapshotBytes != memorycontract.MaximumSnapshotBytes || len(fixture.Cases) == 0 {
		t.Fatal("invalid fixture limits or empty corpus")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			payload := []byte(c.SnapshotJSON)
			var snapshot snapshotDocument
			err := strictjson.Decode(payload, &snapshot)
			if err == nil {
				names := make([]string, len(snapshot.Notes))
				for i, note := range snapshot.Notes {
					names[i] = note.Name
				}
				selection, optionErr := memorycontract.SnapshotOptions(map[string]any{"namespace": snapshot.Namespace, "notes": strings.Join(names, ",")})
				err = optionErr
				if err == nil {
					err = validateSnapshot(payload, selection)
				}
			}
			if (err == nil) != c.SnapshotValid {
				t.Fatalf("snapshot valid=%t, want %t: %v", err == nil, c.SnapshotValid, err)
			}
		})
	}
}
