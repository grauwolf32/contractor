package public

import "testing"

func TestDecodeStrictWithPresenceSeparatesAbsentFromNull(t *testing.T) {
	var target struct {
		Name  *string `json:"name"`
		Label *string `json:"label"`
	}
	fields, err := decodeStrictWithPresence([]byte(`{"name":"a","label":null}`), &target)
	if err != nil || target.Name == nil || *target.Name != "a" {
		t.Fatalf("decodeStrictWithPresence = %v, %+v", err, target)
	}
	if fields.null("name") || !fields.null("label") || fields.null("other") {
		t.Fatalf("null() = %v %v %v", fields.null("name"), fields.null("label"), fields.null("other"))
	}
	if fields.missingOrNull("name") || !fields.missingOrNull("label") || !fields.missingOrNull("other") {
		t.Fatal("missingOrNull() misclassified a member")
	}
	for _, input := range []string{`{"unknown":1}`, `{"name":"a"} {}`, `[1]`} {
		if _, err := decodeStrictWithPresence([]byte(input), &target); err == nil {
			t.Fatalf("decodeStrictWithPresence(%s) accepted invalid input", input)
		}
	}
	fields, err = decodeStrictWithPresence([]byte(`null`), &target)
	if err != nil || fields != nil {
		t.Fatalf("decodeStrictWithPresence(null) = %v, %v", fields, err)
	}
}
