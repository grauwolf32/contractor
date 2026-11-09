package findingintake

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedFindingCandidateEnvelope(t *testing.T) {
	raw := contractstest.ReadFile(t, "api", "testdata", "finding-candidates-cases.json")
	var fixture struct {
		CandidateLimit int `json:"candidateLimit"`
		BatchBytes     int `json:"batchByteLimit"`
		EvidenceBytes  int `json:"evidenceByteLimit"`
		Cases          []struct {
			Name  string          `json:"name"`
			Valid bool            `json:"valid"`
			Batch json.RawMessage `json:"batch"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.CandidateLimit != MaxCandidateBatch || fixture.BatchBytes != MaxCandidateBatchBytes || fixture.EvidenceBytes != MaxCandidateBatchBytes {
		t.Fatal("cross-language limits differ")
	}
	for _, item := range fixture.Cases {
		t.Run(item.Name, func(t *testing.T) {
			_, err := decodeCandidateBatch(item.Batch)
			if (err == nil) != item.Valid {
				t.Fatalf("valid=%t error=%v", item.Valid, err)
			}
		})
	}
}

func TestSharedCandidateEnvelopeExactAndOneOverBounds(t *testing.T) {
	raw := contractstest.ReadFile(t, "api", "testdata", "finding-candidates-cases.json")
	var fixture struct {
		Cases []struct {
			Batch json.RawMessage `json:"batch"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	batch, err := decodeCandidateBatch(fixture.Cases[0].Batch)
	if err != nil {
		t.Fatal(err)
	}
	first := batch.Candidates[0]
	for _, count := range []int{MaxCandidateBatch, MaxCandidateBatch + 1} {
		batch.Candidates = nil
		for index := 0; index < count; index++ {
			candidate := first
			candidate.Key = fmt.Sprintf("%064x", index)
			batch.Candidates = append(batch.Candidates, candidate)
		}
		data, _ := json.Marshal(batch)
		_, err := decodeCandidateBatch(data)
		if (err == nil) != (count == MaxCandidateBatch) {
			t.Fatalf("count=%d err=%v", count, err)
		}
	}
	data := fixture.Cases[0].Batch
	padded := append(append([]byte{}, data...), bytes.Repeat([]byte(" "), MaxCandidateBatchBytes-len(data))...)
	if _, err := decodeCandidateBatch(padded); err != nil {
		t.Fatal("exact byte limit rejected", err)
	}
	if _, err := decodeCandidateBatch(append(padded, ' ')); err == nil {
		t.Fatal("one-over byte limit accepted")
	}
}
