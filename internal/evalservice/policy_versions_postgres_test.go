package evalservice

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestSharedPostgresPreparedBindingsPinSummarizerUsageNormalizer(t *testing.T) {
	h := newHarness(t)
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	tick(t, h.coordinator(t, "normalizer-preparation"))
	e = h.get(t, e.ID)
	if e.State != evaldomain.StateReady {
		t.Fatalf("preparation did not settle: %s", e.State)
	}
	store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
	plan, err := store.FrozenPlan(t.Context(), e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Variants []variantReference `json:"variants"`
	}
	if err := json.Unmarshal(plan.Document.Bytes(), &document); err != nil || len(document.Variants) != 2 {
		t.Fatalf("prepared variants: %+v %v", document, err)
	}
	for _, variant := range document.Variants {
		resource, err := store.PlanResource(t.Context(), e.OwnerID, e.ID, variant.Binding.Resource)
		if err != nil {
			t.Fatal(err)
		}
		var binding portableBinding
		if err := json.Unmarshal(resource.Document.Bytes(), &binding); err != nil {
			t.Fatal(err)
		}
		if len(binding.Normalizers) != 1 || binding.Normalizers[0].ID != "contractor-output@4" || binding.Normalizers[0].ImplementationSHA256 != NormalizerSHA256() {
			t.Fatalf("wrong frozen normalizer: %+v", binding.Normalizers)
		}
	}
}
