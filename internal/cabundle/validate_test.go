package cabundle

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestSharedGrammarCases runs the case table that Runtime contract tests also
// consume, so both validators accept and reject exactly the same bundles.
func TestSharedGrammarCases(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "ca-bundle-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
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
