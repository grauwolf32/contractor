package contracts_test

import (
	"bytes"
	"encoding/json"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// TestSharedCredentialFieldValueCases holds contracts.ValidCredentialFieldValue,
// RuntimeSettings decoding and the private allocation schema, with ECMA-262
// patterns, to api/testdata/v1alpha1/credential-field-value-cases.json. The
// Runtime checks the same cases (runtime/tests/test_shared_runtime_settings_bounds.py),
// and the Runtime credential constructors and public create request are
// checked in internal/credentials and internal/httpapi/public.
func TestSharedCredentialFieldValueCases(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadCredentialFieldValueCases(t)
	for _, test := range cases.Grammar {
		if got := contracts.ValidCredentialFieldValue(test.Value); got != test.Valid {
			t.Errorf("%s: ValidCredentialFieldValue(%q) = %t, want %t", test.Name, test.Value, got, test.Valid)
		}
	}

	compiler, ids := privateSchemaCompiler(t)
	schema, err := compiler.Compile(ids["allocation.schema.json"] + "#/$defs/RuntimeSettings")
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, field := range cases.Fields {
		if field.RuntimeSettings == nil {
			continue
		}
		checked++
		t.Run(field.Name, func(t *testing.T) {
			for _, test := range field.Cases(cases.Grammar) {
				data := credentialFieldValueSettings(t, cases.Base, *field.RuntimeSettings, test.Value)
				if _, err := contracts.DecodePrivateStrict[runtimesettings.RuntimeSettings](data); (err == nil) != test.Valid {
					t.Errorf("%s: RuntimeSettings accepted = %t, want %t: %v", test.Name, err == nil, test.Valid, err)
				}
				document, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
				if err != nil {
					t.Fatal(err)
				}
				if err := schema.Validate(document); (err == nil) != test.Valid {
					t.Errorf("%s: private schema accepted = %t, want %t", test.Name, err == nil, test.Valid)
				}
			}
		})
	}
	if checked == 0 {
		t.Fatal("no shared credential field value names a RuntimeSettings location")
	}
}

// credentialFieldValueSettings returns the base RuntimeSettings document with
// the location's set members applied and value placed at its pointer.
func credentialFieldValueSettings(
	t *testing.T,
	base []byte,
	location contractstest.CredentialFieldValueSettings,
	value string,
) []byte {
	t.Helper()
	var document map[string]any
	if err := json.Unmarshal(base, &document); err != nil {
		t.Fatal(err)
	}
	for pointer, raw := range location.Set {
		var member any
		if err := json.Unmarshal(raw, &member); err != nil {
			t.Fatal(err)
		}
		contractstest.SetJSONPointer(t, document, pointer, member)
	}
	contractstest.SetJSONPointer(t, document, location.Pointer, value)
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
