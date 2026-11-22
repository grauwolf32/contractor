package projectsitemap

import (
	"encoding/json"
	"os"
	"testing"
)

func TestSharedManualEditorIdentityFixture(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/sitemap-editor-source-id.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct{ Namespace, Deployment, SourceID string }
	if err = json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 4 {
		t.Fatal("missing identity cases")
	}
	for _, c := range cases {
		if actual := key("manual", c.Deployment, c.Namespace); actual != c.SourceID {
			t.Fatalf("manual editor identity=%s want=%s", actual, c.SourceID)
		}
	}
}
