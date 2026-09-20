package auditpriority

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

func verdictFixture() Verdict {
	return Verdict{
		ItemKey: "AUTHZ-07", Priority: PriorityHigh, Confidence: ConfidenceMedium,
		Rationale:   "Existing cross-tenant access reports make this check useful.",
		EvidenceIDs: []string{"finding-3"}, MissingContext: []string{"The tenant isolation design is not supplied."},
	}
}

func verdictJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func requireInvalidVerdict(t *testing.T, err error) {
	t.Helper()
	var typed *Error
	if !errors.As(err, &typed) || typed.Code != CodeInvalidVerdict || err.Error() != "audit priority: priority_invalid_verdict" {
		t.Fatalf("expected closed verdict diagnostic, got %v", err)
	}
}

func TestVerdictSchemaFixtures(t *testing.T) {
	base := "../../api/audit-priority/v1"
	path, err := filepath.Abs(filepath.Join(base, "verdict.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	schema, err := jsonschema.NewCompiler().Compile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, category := range []string{"valid", "invalid"} {
		files, err := filepath.Glob(filepath.Join(base, "fixtures", category, "*.json"))
		if err != nil || len(files) == 0 {
			t.Fatalf("fixtures %s: %v", category, err)
		}
		for _, file := range files {
			t.Run(category+"/"+filepath.Base(file), func(t *testing.T) {
				data, err := os.ReadFile(file)
				if err != nil {
					t.Fatal(err)
				}
				value, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
				if err != nil {
					t.Fatal(err)
				}
				schemaError := schema.Validate(value)
				verdict, decodeError := DecodeVerdict(data, "AUTHZ-07", []string{"finding-3"})
				if category == "invalid" {
					if schemaError == nil {
						t.Fatal("schema accepted invalid fixture")
					}
					requireInvalidVerdict(t, decodeError)
					return
				}
				if schemaError != nil || decodeError != nil {
					t.Fatalf("valid fixture rejected: schema %v, decoder %v", schemaError, decodeError)
				}
				canonical, err := MarshalVerdict(verdict)
				if err != nil || !bytes.Equal(canonical, bytes.TrimSpace(data)) {
					t.Fatalf("canonical fixture differs: %v", err)
				}
			})
		}
	}
}

func TestDecodeVerdictStrictFieldsAndSyntax(t *testing.T) {
	valid := string(verdictJSON(t, verdictFixture()))
	cases := map[string]string{
		"duplicate":         strings.Replace(valid, `"priority":"high"`, `"priority":"high","priority":"low"`, 1),
		"escaped duplicate": strings.Replace(valid, `"priority":"high"`, `"priority":"high","\u0070riority":"high"`, 1),
		"unknown secret":    strings.TrimSuffix(valid, "}") + `,"private":"do-not-leak"}`,
		"case alias":        strings.Replace(valid, `"item_key"`, `"ITEM_KEY"`, 1),
		"two objects":       valid + valid,
		"trailing null":     valid + " null",
		"trailing comma":    strings.TrimSuffix(valid, "}") + ",}",
		"truncated":         valid[:len(valid)-1],
		"empty":             "",
		"null object":       "null",
		"array object":      "[" + valid + "]",
		"nested scalar":     strings.Replace(valid, `"confidence":"medium"`, `"confidence":{"private":"do-not-leak"}`, 1),
		"nested evidence":   strings.Replace(valid, `["finding-3"]`, `[["finding-3"]]`, 1),
		"deep object":       `{"rationale":` + strings.Repeat("[", 3000) + strings.Repeat("]", 3000) + `}`,
	}
	for _, key := range []string{"item_key", "priority", "confidence", "rationale", "evidence_ids", "missing_context"} {
		var object map[string]any
		if err := json.Unmarshal([]byte(valid), &object); err != nil {
			t.Fatal(err)
		}
		object[key] = nil
		cases["null "+key] = string(verdictJSON(t, object))
		delete(object, key)
		cases["missing "+key] = string(verdictJSON(t, object))
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			value, err := DecodeVerdict([]byte(data), "AUTHZ-07", []string{"finding-3"})
			requireInvalidVerdict(t, err)
			if !reflect.DeepEqual(value, Verdict{}) {
				t.Fatal("invalid response returned partial model data")
			}
		})
	}
	// Escaped spelling of an exact key is still the same JSON member.
	escapedKey := strings.Replace(valid, `"priority"`, `"\u0070riority"`, 1)
	if _, err := DecodeVerdict([]byte(escapedKey+" \n\t"), "AUTHZ-07", []string{"finding-3"}); err != nil {
		t.Fatalf("valid escaped key/trailing whitespace rejected: %v", err)
	}
}

func TestVerdictUnicodeValidation(t *testing.T) {
	valid := verdictFixture()
	valid.Rationale = "UNICODE_SENTINEL"
	wire := string(verdictJSON(t, valid))
	invalid := []string{
		`\ud800`, `\udc00`, `\ud800\u0061`, `\ud800x\udc00`,
		`\ud800\ud800`, `\uDFFF`, `\ud83d\ud`, `\uxxxx`,
		string([]byte{0xff}), string([]byte{0xed, 0xa0, 0x80}), `reason\u0000suffix`,
	}
	for index, raw := range invalid {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			_, err := DecodeVerdict([]byte(strings.Replace(wire, "UNICODE_SENTINEL", raw, 1)), "AUTHZ-07", []string{"finding-3"})
			requireInvalidVerdict(t, err)
		})
	}
	for _, raw := range []string{`\ud83d\udd10`, `\uD83D\uDD10`, `\ufffd`, "�", "Привет 🔐", `literal \\ud800`, `line\n\ttext`, `quoted \"text\"`} {
		got, err := DecodeVerdict([]byte(strings.Replace(wire, "UNICODE_SENTINEL", raw, 1)), "AUTHZ-07", []string{"finding-3"})
		if err != nil {
			t.Fatalf("valid Unicode %q rejected: %v", raw, err)
		}
		canonical, err := MarshalVerdict(got)
		if err != nil {
			t.Fatal(err)
		}
		again, err := DecodeVerdict(canonical, "AUTHZ-07", []string{"finding-3"})
		if err != nil || !reflect.DeepEqual(got, again) {
			t.Fatalf("Unicode round trip differs: %v", err)
		}
	}
}

func TestVerdictIdentityAndContextAuthority(t *testing.T) {
	valid := verdictFixture()
	cases := []struct {
		name string
		key  string
		ids  []string
	}{
		{"foreign item", "AUTHZ-08", []string{"finding-3"}},
		{"invalid assigned key", "AUTHZ/07", []string{"finding-3"}},
		{"foreign evidence", valid.ItemKey, []string{"finding-4"}},
		{"no context evidence", valid.ItemKey, nil},
		{"duplicate context", valid.ItemKey, []string{"finding-3", "finding-3"}},
		{"invalid context ID", valid.ItemKey, []string{"finding-3", "secret space"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			requireInvalidVerdict(t, ValidateVerdict(valid, tc.key, tc.ids))
			_, err := DecodeVerdict(verdictJSON(t, valid), tc.key, tc.ids)
			requireInvalidVerdict(t, err)
		})
	}
	ids := make([]string, MaxContextEvidence)
	for i := range ids {
		ids[i] = fmt.Sprintf("finding-%d", i)
	}
	if err := ValidateVerdict(valid, valid.ItemKey, ids); err != nil {
		t.Fatal(err)
	}
	requireInvalidVerdict(t, ValidateVerdict(valid, valid.ItemKey, append(ids, "extra")))
	valid.EvidenceIDs = []string{}
	valid.MissingContext = []string{}
	if err := ValidateVerdict(valid, valid.ItemKey, nil); err != nil {
		t.Fatalf("empty context and explicit empty arrays must work: %v", err)
	}
	if _, err := MarshalVerdict(verdictFixture()); err != nil {
		t.Fatalf("structural marshaling must not invent external membership authority: %v", err)
	}
}

func TestVerdictFieldBounds(t *testing.T) {
	for name, mutate := range map[string]func(*Verdict){
		"empty key":          func(v *Verdict) { v.ItemKey = "" },
		"long key":           func(v *Verdict) { v.ItemKey = strings.Repeat("x", MaxIdentifierBytes+1) },
		"non ASCII key":      func(v *Verdict) { v.ItemKey = "АUTHZ-07" },
		"invalid UTF8 text":  func(v *Verdict) { v.Rationale = string([]byte{0xff}) },
		"empty text":         func(v *Verdict) { v.Rationale = "" },
		"unicode whitespace": func(v *Verdict) { v.Rationale = "\u2003\u3000\t" },
		"rationale overflow": func(v *Verdict) { v.Rationale = strings.Repeat("界", 682) + "abc" },
		"nil evidence":       func(v *Verdict) { v.EvidenceIDs = nil },
		"nil missing":        func(v *Verdict) { v.MissingContext = nil },
		"duplicate evidence": func(v *Verdict) { v.EvidenceIDs = []string{"finding-3", "finding-3"} },
		"long evidence":      func(v *Verdict) { v.EvidenceIDs = []string{strings.Repeat("x", MaxIdentifierBytes+1)} },
		"too many evidence": func(v *Verdict) {
			for i := 0; i < MaxEvidenceIDs; i++ {
				v.EvidenceIDs = append(v.EvidenceIDs, fmt.Sprintf("extra-%d", i))
			}
		},
		"too many missing": func(v *Verdict) {
			for i := 0; i < MaxMissingContextEntries; i++ {
				v.MissingContext = append(v.MissingContext, fmt.Sprintf("Missing %d", i))
			}
		},
		"duplicate missing":  func(v *Verdict) { v.MissingContext = []string{"Absent", "Absent"} },
		"blank missing":      func(v *Verdict) { v.MissingContext = []string{" \u00a0"} },
		"missing overflow":   func(v *Verdict) { v.MissingContext = []string{strings.Repeat("界", 85) + "ab"} },
		"invalid priority":   func(v *Verdict) { v.Priority = "HIGH" },
		"invalid confidence": func(v *Verdict) { v.Confidence = "critical" },
	} {
		t.Run(name, func(t *testing.T) {
			value := verdictFixture()
			mutate(&value)
			data, err := MarshalVerdict(value)
			requireInvalidVerdict(t, err)
			if data != nil {
				t.Fatal("invalid verdict returned encoded data")
			}
		})
	}
	valid := verdictFixture()
	valid.ItemKey = strings.Repeat("k", MaxIdentifierBytes)
	valid.Rationale = strings.Repeat("界", 682) + "ab"
	valid.MissingContext = []string{strings.Repeat("界", 85) + "a"}
	valid.EvidenceIDs = []string{strings.Repeat("e", MaxIdentifierBytes)}
	if err := ValidateVerdict(valid, valid.ItemKey, valid.EvidenceIDs); err != nil {
		t.Fatalf("exact UTF-8 byte and ID limits rejected: %v", err)
	}
}

func TestVerdictRawAndCanonicalSizeBounds(t *testing.T) {
	value := verdictFixture()
	value.EvidenceIDs = []string{}
	value.MissingContext = []string{}
	for i := 0; i < 16; i++ {
		value.EvidenceIDs = append(value.EvidenceIDs, fmt.Sprintf("%03d", i)+strings.Repeat("e", MaxIdentifierBytes-3))
		value.MissingContext = append(value.MissingContext, fmt.Sprintf("%03d", i)+strings.Repeat("m", MaxMissingContextBytes-3))
	}
	value.Rationale = "r"
	remaining := MaxVerdictBytes - len(verdictJSON(t, value)) + 1
	if remaining < 1 || remaining > MaxRationaleBytes {
		t.Fatalf("invalid boundary fixture padding: %d", remaining)
	}
	value.Rationale = strings.Repeat("r", remaining)
	canonical, err := MarshalVerdict(value)
	if err != nil || len(canonical) != MaxVerdictBytes {
		t.Fatalf("exact canonical size rejected: length=%d error=%v", len(canonical), err)
	}
	if _, err := DecodeVerdict(canonical, value.ItemKey, value.EvidenceIDs); err != nil {
		t.Fatalf("exact raw size rejected: %v", err)
	}
	_, err = DecodeVerdict(append(canonical, ' '), value.ItemKey, value.EvidenceIDs)
	requireInvalidVerdict(t, err)
	value.Rationale += "r"
	requireInvalidVerdict(t, ValidateVerdict(value, value.ItemKey, value.EvidenceIDs))
	_, err = MarshalVerdict(value)
	requireInvalidVerdict(t, err)
	// JSON escaping can exceed the response limit while decoded field bytes fit.
	value = verdictFixture()
	value.Rationale = strings.Repeat("\x01", MaxRationaleBytes)
	_, err = MarshalVerdict(value)
	requireInvalidVerdict(t, err)
}

func TestCloneVerdictDetachesArrays(t *testing.T) {
	original := verdictFixture()
	clone := cloneVerdict(original)
	clone.EvidenceIDs[0] = "other"
	clone.MissingContext[0] = "different"
	if original.EvidenceIDs[0] != "finding-3" || original.MissingContext[0] != "The tenant isolation design is not supplied." {
		t.Fatal("clone aliases caller input")
	}
	empty := cloneVerdict(Verdict{EvidenceIDs: []string{}, MissingContext: []string{}})
	if empty.EvidenceIDs == nil || empty.MissingContext == nil {
		t.Fatal("clone changed required empty arrays into null")
	}
}

func FuzzDecodeVerdict(f *testing.F) {
	for _, raw := range []string{
		`{"item_key":"AUTHZ-07","priority":"medium","confidence":"low","rationale":"Baseline","evidence_ids":[],"missing_context":[]}`,
		`{"rationale":"\ud800"}`, `{"rationale":[[[[]]]]}`, `null`, "\xff", "{}{}",
	} {
		f.Add([]byte(raw))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		value, err := DecodeVerdict(data, "AUTHZ-07", []string{"finding-3"})
		if err != nil {
			requireInvalidVerdict(t, err)
			return
		}
		canonical, err := MarshalVerdict(value)
		if err != nil || len(canonical) > MaxVerdictBytes {
			t.Fatalf("accepted verdict cannot be encoded within bounds: %v", err)
		}
		again, err := DecodeVerdict(canonical, "AUTHZ-07", []string{"finding-3"})
		if err != nil || !reflect.DeepEqual(value, again) {
			t.Fatalf("accepted verdict changed on canonical round trip: %v", err)
		}
	})
}
