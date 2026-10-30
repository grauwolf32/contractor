package memory

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedMemoryProjectionJSON(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "memory", "projections.json")
	var cases []struct {
		ID      string         `json:"id"`
		Note    StoredNote     `json:"note"`
		Full    map[string]any `json:"full"`
		Preview map[string]any `json:"preview"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	created := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	for _, fixture := range cases {
		t.Run(fixture.ID, func(t *testing.T) {
			note, err := Normalize(fixture.Note)
			if err != nil {
				t.Fatal(err)
			}
			full := FullProjection(note, created, created.Add(time.Second))
			preview := PreviewProjection(note, created, created.Add(time.Second))
			for _, item := range []struct {
				value any
				want  map[string]any
			}{{full, fixture.Full}, {preview, fixture.Preview}} {
				encoded, err := json.Marshal(item.value)
				if err != nil {
					t.Fatal(err)
				}
				var got map[string]any
				if err := json.Unmarshal(encoded, &got); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, item.want) {
					t.Fatalf("projection=%s want=%+v", encoded, item.want)
				}
			}
			if len(full.Tags) > 0 {
				full.Tags[0] = "changed"
				preview.Tags[0] = "changed"
				if note.Tags[0] != "architecture" {
					t.Fatal("projection aliases stored tags")
				}
			}
		})
	}
}
