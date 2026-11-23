package public

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestCapabilitiesPageReusesSnapshotAndPreservesViews(t *testing.T) {
	root := configtest.CopyWithPolicies(t, "../../../testdata/configs")
	manifest, err := os.ReadFile(filepath.Join(root, "workflows", "artifact_copy.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const added = 500
	for i := range added {
		name := fmt.Sprintf("capability-%04d", i)
		content := strings.Replace(string(manifest), "name: artifact-copy\n", "name: "+name+"\n", 1)
		if content == string(manifest) {
			t.Fatal("Workflow fixture identity changed")
		}
		if err := os.WriteFile(filepath.Join(root, "workflows", name+".yaml"), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	managed := t.TempDir()
	snapshot, _, err := configload.LoadUnionReadOnly(root, managed)
	if err != nil {
		t.Fatal(err)
	}
	var current atomic.Pointer[config.Snapshot]
	current.Store(snapshot)
	reader := &capabilitiesReader{}
	for _, kind := range []string{"", "workflow", "audit"} {
		var rows []capabilityBinding
		after, digest := "", ""
		for {
			page, err := reader.bindings(current.Load(), kind, after, 7, digest)
			if err != nil {
				t.Fatal(err)
			}
			if len(page.Items) > 7 {
				t.Fatal("page exceeded requested rows")
			}
			for _, row := range page.Items {
				key := row.Kind + ":" + row.Selector
				if key <= after {
					t.Fatal("capability order repeated or regressed")
				}
				after = key
			}
			rows = append(rows, page.Items...)
			digest = page.Snapshot
			if !page.HasMore {
				break
			}
			if page.LastKey != after {
				t.Fatal("page position does not match last visible binding")
			}
		}
		expected := []capabilityBinding{}
		if kind != "audit" {
			for _, workflow := range snapshot.Workflows() {
				expected = append(expected, capabilityBinding{Kind: "workflow", Selector: workflow.Ref.Name + "@" + workflow.Ref.Version, Available: true})
			}
		}
		if kind != "workflow" {
			for _, profile := range snapshot.AuditProfiles() {
				expected = append(expected, capabilityBinding{Kind: "audit", Selector: profile.Ref.Name + "@" + profile.Ref.Version, Available: true})
			}
		}
		// Independently reproduce the former HTTP response projection, including
		// null reasons, sorting, kind-specific hashes and the exact binding set.
		sortBindings(expected)
		raw, _ := json.Marshal(expected)
		got, _ := json.Marshal(rows)
		if len(expected) == 0 {
			got = []byte("[]")
		}
		if string(got) != string(raw) || digest != evaldomain.Digest(raw) {
			t.Fatalf("kind %q changed bindings or view fingerprint", kind)
		}
	}
	allocations := testing.AllocsPerRun(20, func() {
		page, err := reader.bindings(current.Load(), "workflow", "", 1, "")
		if err != nil || len(page.Items) != 1 {
			panic("capability page failed")
		}
	})
	if allocations > 20 {
		t.Fatalf("one cached binding allocated %.0f objects for %d extra Workflows", allocations, added)
	}
	t.Logf("one binding from %d Workflows uses %.0f allocations", snapshot.Counts().Workflows, allocations)
	first, err := reader.bindings(current.Load(), "workflow", "", 1, "")
	if err != nil {
		t.Fatal(err)
	}
	// A new snapshot with identical bindings keeps the old cursor valid.
	unchanged, _, err := configload.LoadUnionReadOnly(root, managed)
	if err != nil {
		t.Fatal(err)
	}
	current.Store(unchanged)
	if _, err := reader.bindings(current.Load(), "workflow", first.LastKey, 1, first.Snapshot); err != nil {
		t.Fatalf("unchanged view rejected cursor: %v", err)
	}
	originalSelector := first.Items[0].Selector
	first.Items[0].Selector = "caller mutation"
	detached, err := reader.bindings(current.Load(), "workflow", "", 1, "")
	if err != nil || detached.Items[0].Selector != originalSelector {
		t.Fatal("page mutation reached catalog cache")
	}
	changed := strings.Replace(string(manifest), "name: artifact-copy\n", "name: capability-new\n", 1)
	if err := os.WriteFile(filepath.Join(root, "workflows", "new.yaml"), []byte(changed), 0o644); err != nil {
		t.Fatal(err)
	}
	replacement, _, err := configload.LoadUnionReadOnly(root, managed)
	if err != nil {
		t.Fatal(err)
	}
	current.Store(replacement)
	if _, err := reader.bindings(current.Load(), "workflow", first.LastKey, 1, first.Snapshot); !evaldomain.IsCode(err, "eval_view_changed") {
		t.Fatalf("replaced catalog cursor = %v", err)
	}
	if _, err := reader.bindings(current.Load(), "workflow", "", 1, ""); err != nil {
		t.Fatal(err)
	}
}

func sortBindings(rows []capabilityBinding) {
	// Keep ordering equivalent to the public binding key, independent of caches.
	slices.SortFunc(rows, func(a, b capabilityBinding) int { return strings.Compare(a.Kind+":"+a.Selector, b.Kind+":"+b.Selector) })
}
