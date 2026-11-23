package engagementcontract_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

type fixture struct {
	Documents []struct {
		Name      string `json:"name"`
		Text      string `json:"text"`
		Valid     bool   `json:"valid"`
		Schema    string `json:"schema"`
		Canonical string `json:"canonical"`
		Digest    string `json:"digest"`
	} `json:"documents"`
	URLs []struct {
		URL       string `json:"url"`
		Canonical string `json:"canonical"`
	} `json:"urls"`
}

func TestSharedEngagement(t *testing.T) {
	var f fixture
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "engagement-cases.json"), &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Documents) == 0 || len(f.URLs) == 0 {
		t.Fatal("fixture selections must be nonempty")
	}
	c := jsonschema.NewCompiler()
	schemaData := contractstest.ReadFile(t, "api", "v1alpha1", "engagement.schema.json")
	source, err := jsonschema.UnmarshalJSON(bytes.NewReader(schemaData))
	if err != nil {
		t.Fatal(err)
	}
	id := source.(map[string]any)["$id"].(string)
	if err = c.AddResource(id, source); err != nil {
		t.Fatal(err)
	}
	schema, err := c.Compile(id)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range f.Documents {
		t.Run(tc.Name, func(t *testing.T) {
			doc, err := engagementcontract.Decode([]byte(tc.Text))
			if (err == nil) != tc.Valid {
				t.Fatalf("accepted=%v want=%v (%v)", err == nil, tc.Valid, err)
			}
			if err == nil {
				canonical, e := strictjson.Canonical(doc)
				if e != nil || string(canonical) != tc.Canonical {
					t.Fatal("canonical encoding differs", e)
				}
				digest := contentdigest.Bytes(canonical)
				if digest != tc.Digest {
					t.Fatal("canonical digest differs", e)
				}
			}
			value, e := jsonschema.UnmarshalJSON(strings.NewReader(tc.Text))
			accepted := e == nil && schema.Validate(value) == nil
			want := tc.Valid || tc.Schema == "accepts"
			if accepted != want {
				t.Errorf("schema accepted=%v want=%v", accepted, want)
			}
		})
	}
	for _, tc := range f.URLs {
		t.Run("url="+tc.URL, func(t *testing.T) {
			value, err := engagementcontract.CanonicalURL(tc.URL)
			if tc.Canonical == "" {
				if err == nil {
					t.Fatal("accepted invalid URL")
				}
			} else if err != nil || value != tc.Canonical {
				t.Fatalf("canonical=%q err=%v want=%q", value, err, tc.Canonical)
			}
		})
	}
}
func TestEngagementDecodeBoundsInputBeforeParsing(t *testing.T) {
	if _, err := engagementcontract.Decode(bytes.Repeat([]byte(" "), engagementcontract.MaxDocumentBytes+1)); err == nil {
		t.Fatal("unbounded document accepted")
	}
}
