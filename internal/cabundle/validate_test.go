package cabundle

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedGrammarCases runs the case table that Runtime contract tests also
// consume, so both validators accept and reject exactly the same bundles.
func TestSharedGrammarCases(t *testing.T) {
	t.Parallel()

	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "ca-bundle-cases.json")
	type grammarCase struct {
		Name  string `json:"name"`
		Value string `json:"value"`
	}
	var cases struct {
		Valid   []grammarCase `json:"valid"`
		Invalid []grammarCase `json:"invalid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Valid) == 0 || len(cases.Invalid) == 0 {
		t.Fatal("shared CA bundle case table is empty")
	}
	for _, test := range cases.Valid {
		t.Run("valid/"+test.Name, func(t *testing.T) {
			t.Parallel()
			if err := Validate(test.Value); err != nil {
				t.Fatalf("valid CA bundle rejected: %v", err)
			}
		})
	}
	for _, test := range cases.Invalid {
		t.Run("invalid/"+test.Name, func(t *testing.T) {
			t.Parallel()
			if err := Validate(test.Value); !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid CA bundle error = %v, want ErrInvalid", err)
			}
		})
	}
	single := cases.Valid[0].Value
	if err := Validate(strings.Repeat(single, 8)); err != nil {
		t.Fatalf("eight certificates rejected: %v", err)
	}
	if err := Validate(strings.Repeat(single, 9)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("nine certificates error = %v, want ErrInvalid", err)
	}
}

// TestSharedBundlesAreASCII pins the ASCII grammar that the public CABundlePEM
// schema declares (LIM-RCF-11): a valid bundle with any shared non-ASCII
// character before, inside or after its block is rejected, so maxLength
// counts its bytes.
func TestSharedBundlesAreASCII(t *testing.T) {
	t.Parallel()
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "ca-bundle-cases.json")
	var cases struct {
		Valid []struct {
			Value string `json:"value"`
		} `json:"valid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil || len(cases.Valid) == 0 {
		t.Fatalf("shared CA bundle cases: %v", err)
	}
	bundle := cases.Valid[0].Value
	characters, _ := contractstest.ReadUTF8ByteBoundCases(t)
	for _, character := range characters {
		if len(character) == 1 {
			continue
		}
		middle := strings.Index(bundle, "\n-----END")
		for name, value := range map[string]string{
			"before": character + "\n" + bundle,
			"inside": bundle[:middle] + character + bundle[middle:],
			"after":  bundle + character + "\n",
		} {
			if err := Validate(value); !errors.Is(err, ErrInvalid) {
				t.Errorf("%q %s the block: error = %v, want ErrInvalid", character, name, err)
			}
		}
	}
}
