package auditdomain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

type dependencyCases struct {
	CanonicalDigest    string          `json:"canonicalDigest"`
	MaximumDescriptors int             `json:"maximumDescriptors"`
	MaximumBytes       int             `json:"maximumBytes"`
	Base               json.RawMessage `json:"base"`
	Cases              []struct {
		Name   string `json:"name"`
		Valid  bool   `json:"valid"`
		Path   []any  `json:"path"`
		Value  any    `json:"value"`
		Delete bool   `json:"delete"`
	} `json:"cases"`
}

func TestSharedRetainedDependencyManifest(t *testing.T) {
	var fixture dependencyCases
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-dependencies", "manifest-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.MaximumDescriptors != MaximumRetainedDependencies || fixture.MaximumBytes != MaximumRetainedDependencyBytes {
		t.Fatal("fixture limits differ")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var root any
			if err := json.Unmarshal(fixture.Base, &root); err != nil {
				t.Fatal(err)
			}
			if len(test.Path) > 0 {
				cursor := root
				for _, part := range test.Path[:len(test.Path)-1] {
					cursor = dependencyPathValue(cursor, part)
				}
				last := test.Path[len(test.Path)-1].(string)
				if test.Delete {
					delete(cursor.(map[string]any), last)
				} else {
					cursor.(map[string]any)[last] = test.Value
				}
			}
			data, err := json.Marshal(root)
			if err != nil {
				t.Fatal(err)
			}
			manifest, err := DecodeRetainedDependencies(data)
			if (err == nil) != test.Valid {
				t.Fatalf("decode valid=%v: %v", test.Valid, err)
			}
			if test.Valid {
				encoded, err := EncodeRetainedDependencies(manifest)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := DecodeRetainedDependencies(encoded); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	base, err := DecodeRetainedDependencies(fixture.Base)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := EncodeRetainedDependencies(base)
	if err != nil || DigestBytes(canonical) != fixture.CanonicalDigest {
		t.Fatalf("canonical retained manifest differs: %v", err)
	}
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(contractstest.Path(t, "api", "audit-dependencies", "v1", "manifest.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(canonical))
	if err != nil || schema.Validate(value) != nil {
		t.Fatalf("valid manifest fails published schema: %v", err)
	}
	for count := MaximumRetainedDependencies; count <= MaximumRetainedDependencies+1; count++ {
		manifest := base
		manifest.Dependencies = make([]RetainedDependency, count)
		for index := range manifest.Dependencies {
			d := base.Dependencies[0]
			d.Name = fmt.Sprintf("dep-%02d", index)
			d.Requires = []string{}
			for predecessor := 0; predecessor < index; predecessor++ {
				d.Requires = append(d.Requires, fmt.Sprintf("dep-%02d", predecessor))
			}
			manifest.Dependencies[index] = d
		}
		if err := ValidateRetainedDependencies(manifest); (err == nil) != (count == MaximumRetainedDependencies) {
			t.Fatalf("count %d: %v", count, err)
		}
	}
	for _, malformed := range [][]byte{
		bytes.Replace(fixture.Base, []byte(`"owner_id": "owner-1"`), []byte(`"owner_id":"owner-1","owner_id":"owner-1"`), 1),
		append(bytes.Clone(fixture.Base), []byte(`{}`)...),
		append(bytes.Clone(fixture.Base), bytes.Repeat([]byte(" "), MaximumRetainedDependencyBytes)...),
	} {
		if _, err := DecodeRetainedDependencies(malformed); err == nil {
			t.Fatal("malformed manifest accepted")
		}
	}
}

func dependencyPathValue(value any, part any) any {
	if name, ok := part.(string); ok {
		return value.(map[string]any)[name]
	}
	return value.([]any)[int(part.(float64))]
}

func TestRetainedDependenciesRebuildExactTaskAndExecutionTogether(t *testing.T) {
	inventory, err := BuildChecklistInventory([]byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"item-1","version":"1","statement":"Review","applicability":"always","allowed_methods":["review"],"required_evidence":[],"review_policy":"automatic"}]}`), JSONMediaType, testInventoryOptions("check"))
	if err != nil {
		t.Fatal(err)
	}
	revision := "revision-1"
	descriptor := testDependencyArtifact(ArtifactNamespace("audit-1"), &revision)
	manifest := RetainedDependencyManifest{Schema: RetainedDependencySchema, OwnerID: "owner-1", ProjectID: "project-1", AuditID: "audit-1", ItemKey: "item-1", BaselineDigest: DigestBytes(nil), ProfileDigest: DigestBytes(nil), WorkflowDigest: DigestBytes(nil), SkillsDigest: DigestBytes(nil), Dependencies: []RetainedDependency{{Name: "baseline", Kind: "baseline", Projection: "exact", Source: descriptor, Retained: descriptor, Origin: RetainedDependencyOrigin{Kind: "baseline", BaselineInput: "source"}, Requires: []string{}}}}
	result, err := WithRetainedDependencies(inventory, map[string]RetainedDependencyManifest{"item-1": manifest})
	if err != nil {
		t.Fatal(err)
	}
	if result.Tasks[0].PackageDigest == inventory.Tasks[0].PackageDigest || result.Worklist.Items[0].TaskPackageID == inventory.Worklist.Items[0].TaskPackageID {
		t.Fatal("dependency bytes did not change deterministic package identity")
	}
	if inventory.Tasks[0].Document.Dependencies != nil {
		t.Fatal("caller inventory changed")
	}
	manifest.Dependencies[0].Name = "mutated"
	if result.Tasks[0].Document.Dependencies.Dependencies[0].Name != "baseline" {
		t.Fatal("caller manifest aliases returned task")
	}
	result.Tasks[0].Document.Dependencies.Dependencies[0].Name = "other"
	if err := ValidateInventory(result); err == nil {
		t.Fatal("task/package dependency disagreement accepted")
	}
	if _, err := WithRetainedDependencies(inventory, map[string]RetainedDependencyManifest{"foreign": manifest}); err == nil {
		t.Fatal("foreign item manifest accepted")
	}
}

func TestRetainedDependencyByteBound(t *testing.T) {
	// Exact revision identities are opaque. Four individually valid long refs
	// prove the independent 256 KiB aggregate bound rather than a field limit.
	revision := strings.Repeat("r", MaximumStringBytes)
	d := testDependencyArtifact(ArtifactNamespace("audit-1"), &revision)
	manifest := RetainedDependencyManifest{Schema: RetainedDependencySchema, OwnerID: "owner", ProjectID: "project", AuditID: "audit-1", ItemKey: "item", BaselineDigest: DigestBytes(nil), ProfileDigest: DigestBytes(nil), WorkflowDigest: DigestBytes(nil), SkillsDigest: DigestBytes(nil), Dependencies: []RetainedDependency{}}
	for index := 0; index < 2; index++ {
		manifest.Dependencies = append(manifest.Dependencies, RetainedDependency{Name: fmt.Sprintf("dep-%d", index), Kind: "baseline", Projection: "exact", Source: d, Retained: d, Origin: RetainedDependencyOrigin{Kind: "baseline", BaselineInput: "source"}, Requires: []string{}})
	}
	if ErrorCode(ValidateRetainedDependencies(manifest)) != CodeLimitExceeded {
		t.Fatal("aggregate byte bound did not reject large valid descriptors")
	}
}

func testDependencyArtifact(namespace string, revision *string) contracts.ExactArtifact {
	return contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: namespace, Name: "source", Revision: revision}, Digest: DigestBytes(nil), MediaType: JSONMediaType, SizeBytes: 0}
}
