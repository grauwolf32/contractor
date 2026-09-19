package evaldomain

import (
	"bytes"
	"encoding/json"
	"errors"
	evalschema "github.com/grauwolf32/contractor/api/evals/v1"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const fixtureDir = "../../api/testdata/evals"

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(fixtureDir, "valid", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return data
}
func objectFixture(t *testing.T, name string) map[string]any {
	t.Helper()
	data, err := StrictJSON(fixture(t, name))
	if err != nil {
		t.Fatal(err)
	}
	return asObject(data)
}

func TestEvalConformanceFixtures(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(fixtureDir, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name, Kind, File, ErrorCode string
		Valid                       bool
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(fixtureDir, c.File))
			if err != nil {
				t.Fatal(err)
			}
			err = Validate(c.Kind, raw)
			if !c.Valid {
				var safe *Error
				if !errors.As(err, &safe) || safe.Code != c.ErrorCode {
					t.Fatalf("want %s; got %v", c.ErrorCode, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			f, err := Freeze(c.Kind, raw)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(raw, f.Bytes()) || f.Digest() != Digest(raw) {
				t.Fatal("retained bytes or hash changed")
			}
			value, err := StrictJSON(raw)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			if err := Validate(c.Kind, encoded); err != nil {
				t.Fatalf("reserialization lost contract: %v", err)
			}
		})
	}
}

func TestEvalTypedAuthoringRoundTrips(t *testing.T) {
	for _, name := range []string{"create-workflow", "create-audit", "external-workflow", "external-audit"} {
		t.Run(name, func(t *testing.T) {
			var typed CreateExperiment
			if err := DecodeInto("CreateExperiment", fixture(t, name), &typed); err != nil {
				t.Fatal(err)
			}
			raw, err := json.Marshal(typed)
			if err != nil {
				t.Fatal(err)
			}
			if err := Validate("CreateExperiment", raw); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEvalPrivateProjectionsAndOwnerErrors(t *testing.T) {
	var input DatasetInput
	if err := DecodeInto("DatasetInput", fixture(t, "dataset"), &input); err != nil {
		t.Fatal(err)
	}
	safe, err := DatasetProjection(input, "evaluation-1", "r1")
	if err != nil {
		t.Fatal(err)
	}
	execution, err := ExecutionProjection(input.Cases[0])
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Freeze("playground.plan/v1", fixture(t, "portable-plan"))
	if err != nil {
		t.Fatal(err)
	}
	projection, err := PublicPlanProjection(plan)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{safe, execution, projection, plan, CheckOwner("owner-1", "OTHER_OWNER_SENTINEL"), Failure("SECRET_CREDENTIAL_SENTINEL")} {
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		for _, secret := range []string{"PRIVATE_TRUTH_SENTINEL", "PRIVATE_RUBRIC_SENTINEL", "SECRET_PATH_SENTINEL", "SECRET_CREDENTIAL_SENTINEL", "OTHER_OWNER_SENTINEL", "/private/oracles"} {
			if bytes.Contains(raw, []byte(secret)) {
				t.Fatalf("private content leaked: %s", secret)
			}
		}
	}
	if projection.SourceRecordSHA256 != plan.Digest() {
		t.Fatal("projection must retain source digest without claiming identical bytes")
	}
	publicBytes, _ := json.Marshal(projection)
	if Digest(publicBytes) == plan.Digest() {
		t.Fatal("projection digest falsely equals private plan")
	}
	input.Cases[0].Task.Parameters["changed"] = "later"
	if _, ok := execution.Task.Parameters["changed"]; ok {
		t.Fatal("execution projection aliases mutable authoring input")
	}
	if CheckOwner("owner-1", "owner-1") != nil {
		t.Fatal("owned resource rejected")
	}
}

func TestEvalStrictJSONBoundsAndAmbiguities(t *testing.T) {
	if err := Validate("Command", bytes.Repeat([]byte(" "), MaxDocumentBytes+1)); err == nil || err.(*Error).Code != "eval_limit_exceeded" {
		t.Fatalf("size bound: %v", err)
	}
	for _, raw := range []string{`{"x":1,"x":2}`, `{"x":{"a":1,"\u0061":2}}`, `{"x":"\ud800"}`, `{"x":"\udc00"}`, `{} {}`, `{"x":NaN}`} {
		if _, err := StrictJSON([]byte(raw)); err == nil {
			t.Fatalf("accepted ambiguous JSON %q", raw)
		}
	}
	for _, raw := range []string{`{"x":"\ud83d\ude00"}`, `{"x":"literal \\ud800"}`, `{"x":"Пример"}`} {
		if _, err := StrictJSON([]byte(raw)); err != nil {
			t.Fatalf("rejected valid Unicode: %v", err)
		}
	}
	bad := append([]byte(`{"x":"`), 0xff)
	bad = append(bad, []byte(`"}`)...)
	if _, err := StrictJSON(bad); err == nil {
		t.Fatal("invalid UTF-8 accepted")
	}
}

func TestEvalHTTPMutationsAndReplay(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(fixtureDir, "http-mutations.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name, Kind, Method, Path string
		Headers                  map[string]string
		Body                     json.RawMessage
		RequiresRevision         bool
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			identity, err := IdentifyMutation(c.Headers["Idempotency-Key"], c.Headers["If-Match"], c.RequiresRevision, c.Kind, c.Body)
			if err != nil {
				t.Fatal(err)
			}
			// Response loss followed by advancing state must still recover this receipt.
			replay, err := CheckMutation(identity, &identity, 99)
			if err != nil || !replay {
				t.Fatalf("exact replay rejected: %v", err)
			}
			changed := identity
			changed.RequestSHA256 = Digest([]byte("changed payload"))
			if _, err := CheckMutation(changed, &identity, 99); err == nil || err.(*Error).Code != "eval_idempotency_conflict" {
				t.Fatalf("changed body accepted: %v", err)
			}
			if c.RequiresRevision {
				if _, err := IdentifyMutation(identity.Key, "", true, c.Kind, c.Body); err == nil || err.(*Error).Status != 428 {
					t.Fatalf("missing CAS accepted: %v", err)
				}
				if _, err := CheckMutation(identity, nil, 99); err == nil || err.(*Error).Status != 412 {
					t.Fatalf("stale CAS accepted: %v", err)
				}
			}
			if !strings.HasPrefix(c.Path, "/v1/") {
				t.Fatal("non-public route in mutation fixture")
			}
		})
	}
	if CheckControlMode("server", "finalize") == nil || CheckControlMode("external", "start") == nil || CheckControlMode("external", "duplicate") == nil {
		t.Fatal("two drivers allowed to control the same experiment")
	}
}

func TestEvalPortableGatesAndIdentity(t *testing.T) {
	var request CreateExperiment
	if err := DecodeInto("CreateExperiment", fixture(t, "create-workflow"), &request); err != nil {
		t.Fatal(err)
	}
	comparison, extensions := PortableComparison(request.Draft.Comparison)
	if _, ok := comparison["gates"]; ok {
		t.Fatal("modified closed portable comparison shape")
	}
	gate := extensions["playground:comparison-gates"].(map[string]any)
	if !reflect.DeepEqual(gate, map[string]any{"min_candidate_end_to_end_pass": float64(1), "max_quality_drop": float64(0)}) {
		t.Fatal(gate)
	}
	var public PublicPlan
	if err := DecodeInto("PublicPlan", fixture(t, "public-plan"), &public); err != nil {
		t.Fatal(err)
	}
	for _, m := range public.Members {
		actual, err := MemberID(public.ExperimentID, m.SuiteID, m.CaseID, m.Sample, m.VariantID)
		if err != nil || actual != m.MemberID {
			t.Fatalf("cross-language identity mismatch: %s %v", actual, err)
		}
	}
	for _, n := range []int{0, -1, 101} {
		if _, err := MemberID("trace-1", "trace-small", "unsafe-query", n, "a"); err == nil {
			t.Fatal("invalid sample accepted")
		}
	}
	if _, err := MemberID("trace-1", "trace-small", "<script>", 1, "a"); err == nil {
		t.Fatal("invalid identity accepted")
	}
}

func TestEvalParameterNamesAreNotResponseSemantics(t *testing.T) {
	value := objectFixture(t, "dataset")
	c := asObject(asRows(value["cases"])[0])
	task := asObject(c["task"])
	task["parameters"] = map[string]any{"denominator": "text", "value": "text", "counts": "text", "conclusion": "text"}
	raw, _ := json.Marshal(value)
	if err := Validate("DatasetInput", raw); err != nil {
		t.Fatal(err)
	}
	if _, err := StrictJSON([]byte(`{"x":1e999}`)); err == nil {
		t.Fatal("unrepresentable number accepted")
	}
}

func TestEvalFrozenBytesCannotBeChangedThroughCallerBuffers(t *testing.T) {
	raw := fixture(t, "portable-plan")
	f, err := Freeze("playground.plan/v1", raw)
	if err != nil {
		t.Fatal(err)
	}
	sha := f.Digest()
	raw[0] = '!'
	copy := f.Bytes()
	copy[0] = '!'
	if Digest(f.Bytes()) != sha {
		t.Fatal("frozen bytes alias caller buffers")
	}
}

func TestEvalRegistrationCannotChangeBindingsOrDropSamples(t *testing.T) {
	v := objectFixture(t, "registration")
	manifest := asObject(v["manifest"])
	members := asRows(manifest["members"])
	asObject(members[2])["binding_sha256"] = Digest([]byte("another binding"))
	raw, _ := json.Marshal(v)
	if err := Validate("ExternalRegistration", raw); err == nil {
		t.Fatal("per-case binding changed within one arm")
	}
	v = objectFixture(t, "registration")
	manifest = asObject(v["manifest"])
	members = asRows(manifest["members"])
	// Remove both arms of sample 2 of one case; retained pairs alone do not
	// establish a complete shared repetition matrix.
	manifest["members"] = append(members[:2], members[4:]...)
	recipes := asRows(v["recipes"])
	v["recipes"] = append(recipes[:2], recipes[4:]...)
	raw, _ = json.Marshal(v)
	if err := Validate("ExternalRegistration", raw); err == nil {
		t.Fatal("missing repeat silently reduced the denominator")
	}
}

func TestEvalPortableSchemaSnapshotMatchesProvenance(t *testing.T) {
	raw, err := evalschema.Files.ReadFile("portable/provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	var provenance struct {
		Repository, Commit string
		Files              map[string]string
	}
	if err := json.Unmarshal(raw, &provenance); err != nil {
		t.Fatal(err)
	}
	if provenance.Repository != "playground-v2" || len(provenance.Commit) != 40 {
		t.Fatal("unpinned portable schema snapshot")
	}
	entries, err := fs.Glob(evalschema.Files, "portable/*.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != len(provenance.Files) {
		t.Fatal("missing schema provenance")
	}
	for _, path := range entries {
		raw, err := evalschema.Files.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if Digest(raw)[7:] != provenance.Files[filepath.Base(path)] {
			t.Fatalf("portable schema bytes changed: %s", path)
		}
	}
	value := objectFixture(t, "portable-plan")
	asObject(value["experiment_ref"])["resource"] = "../private/secret.json"
	raw, _ = json.Marshal(value)
	if err := Validate("playground.plan/v1", raw); err == nil {
		t.Fatal("bundle-relative traversal accepted")
	}
}
