package contracts_test

import (
	"bytes"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// TestSharedTargetCredentialReferenceCases holds the exact target credential
// reference, runtimesettings.RuntimeCredentialRef decoded as private wire
// data, and the private schema's RuntimeCredentialRef to the reference cases
// of api/testdata/v1alpha1/target-credential-cases.json. The Runtime checks
// the same cases (runtime/tests/test_shared_target_credential_references.py).
func TestSharedTargetCredentialReferenceCases(t *testing.T) {
	t.Parallel()
	cases := contractstest.ReadTargetCredentialCases(t)
	compiler, ids := privateSchemaCompiler(t)
	schema, err := compiler.Compile(ids["allocation.schema.json"] + "#/$defs/RuntimeCredentialRef")
	if err != nil {
		t.Fatal(err)
	}
	targetKinds := 0
	for _, test := range cases.References {
		ref, err := contracts.DecodePrivateStrict[runtimesettings.RuntimeCredentialRef](test.Data)
		if (err == nil) != test.Valid {
			t.Errorf("%s: RuntimeCredentialRef accepted = %t, want %t: %v", test.Name, err == nil, test.Valid, err)
		}
		if err == nil {
			switch ref.Kind {
			case runtimesettings.RuntimeCredentialTargetBasic, runtimesettings.RuntimeCredentialTargetBearer,
				runtimesettings.RuntimeCredentialTargetCookies:
				targetKinds++
			}
		}
		accepted := false
		if test.SchemaDocument() {
			document, err := jsonschema.UnmarshalJSON(bytes.NewReader(test.Data))
			if err != nil {
				t.Fatalf("%s: %v", test.Name, err)
			}
			accepted = schema.Validate(document) == nil
		}
		if want := test.Valid || test.SchemaAccepts; accepted != want {
			t.Errorf("%s: private schema accepted = %t, want %t", test.Name, accepted, want)
		}
	}
	if targetKinds == 0 {
		t.Fatal("no valid reference names a target credential kind")
	}
}
