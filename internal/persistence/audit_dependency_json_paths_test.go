package persistence

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// New JSON authority must share its exact typed paths with the strict holder
// and origin readers. A JSON-tag rename must fail before an obligation can
// silently become an unprotected reference.
func TestSharedRetainedDependencySQLReadsSharedDescriptorPaths(t *testing.T) {
	var fixture struct {
		Base json.RawMessage `json:"base"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-dependencies", "manifest-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	manifest, err := auditdomain.DecodeRetainedDependencies(fixture.Base)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	var document any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	migration := allMigrationSQL(t)
	for _, test := range []struct {
		path []string
		sql  string
	}{
		{[]string{"dependencies", "*", "retained", "ref", "revision"}, "contractor_artifact_ref_key('project'::text, contractor_audit_dependency_holds.project, (dependency.dependency #> '{retained,ref}'::text[]))"},
		{[]string{"dependencies", "*", "source", "ref", "revision"}, "dependency #> '{source,ref}'"},
		{[]string{"dependencies", "*", "origin", "kind"}, "dependency #>> '{origin,kind}'"},
		{[]string{"dependencies", "*", "requires"}, "dependency->'requires'"},
		{[]string{"baseline_digest"}, "NEW.retained_dependencies->>'baseline_digest'"},
		{[]string{"owner_id"}, "NEW.retained_dependencies->>'owner_id'"},
	} {
		if len(jsonValuesAt(document, test.path)) == 0 {
			t.Errorf("typed dependency path %v is empty", test.path)
		}
		if !strings.Contains(migration, test.sql) {
			t.Errorf("no registered migration reader uses %q", test.sql)
		}
	}
}
