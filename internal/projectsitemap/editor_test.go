package projectsitemap

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestEditorMutationBindsOnlyTrustedSource(t *testing.T) {
	rule := `{"origin":"https://example.test","method":"GET","path":"/items/{id}","pointer":"","parameters":["id"],"structural":[]}`
	var command EditorMutation
	if err := strictjson.Decode([]byte(`{"operation":"replace-source","expectedRevision":"absent","source":{"rules":[`+rule+`]}}`), &command); err != nil {
		t.Fatal(err)
	}
	mutation := command.WithSource("trusted-source")
	if mutation.Source.SourceID != "trusted-source" || mutation.Source.Rules[0].SourceID != "trusted-source" || mutation.Source.Rules[0].Path != "/items/{id}" {
		t.Fatalf("editor source was not assigned: %+v", mutation.Source)
	}
	for _, body := range []string{
		`{"operation":"replace-source","source":{"sourceId":"forged","rules":[]}}`,
		`{"operation":"replace-source","source":{"rules":[{"sourceId":"forged","origin":"https://example.test","method":"GET","path":"/","pointer":"","parameters":[],"structural":[]}]}}`,
		`{"operation":"observe","observations":[]}`,
		`{"operation":"replace-source","source":{"artifact":{},"rules":[]}}`,
		`{"operation":"replace-source","source":{"rules":null}}`,
		`{"operation":"replace-source","source":null}`,
	} {
		if err := strictjson.Decode([]byte(body), &command); !errors.Is(err, ErrInvalid) {
			t.Fatalf("accepted untrusted editor shape %s: %v", body, err)
		}
	}
	expected := &Source{}
	command.Source = &EditorSourceUpdate{Expected: expected, Rules: []EditorRule{}}
	if command.WithSource("trusted-source").Source.Expected != expected {
		t.Fatal("source precondition was changed")
	}
}
