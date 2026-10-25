package auditdomain

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedWebScanSettings(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "audit-web-scan-settings.json")
	var cases []struct {
		Name     string          `json:"name"`
		Settings json.RawMessage `json:"settings"`
		Valid    bool            `json:"valid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty fixture gate")
	}
	for _, test := range cases {
		t.Run(test.Name, func(t *testing.T) {
			_, err := DecodeWebScanSettings(test.Settings)
			if (err == nil) != test.Valid {
				t.Fatalf("settings valid=%t: %v", test.Valid, err)
			}
		})
	}
}

const webTargetFixture = `{"schema":"contractor.audit.web-target.v1","url":"https://target.test/a?b=1"}`
const webSettingsFixture = `{"schema":"contractor.audit.web-scan-settings.v1","templateIds":"fixture-*","tags":"http","severity":"high,critical","rateLimit":7}`

func TestWebScanInventoryPinsAndReproducesDirectTarget(t *testing.T) {
	options := testInventoryOptions("scan")
	options.ApprovalRequirement = ApprovalActiveCheck
	input := ExactInput{Name: "settings", Ref: options.SourceRef.Clone(), Digest: DigestBytes([]byte(webSettingsFixture))}
	input.Ref.Name = "settings"
	inventory, err := BuildWebScanInventory([]byte(webTargetFixture), JSONMediaType, []byte(webSettingsFixture), input, options)
	if err != nil || len(inventory.Tasks) != 1 || ValidateInventory(inventory) != nil {
		t.Fatalf("inventory: %v", err)
	}
	task := inventory.Tasks[0].Document
	if task.Kind != "web-scan" || task.ExecutionContract() != "web-scan@1" || task.Scan.Operation != "target" || inventory.Worklist.Items[0].ApprovalRequirement != ApprovalActiveCheck {
		t.Fatal("direct target lost its execution or approval identity")
	}
	prepared, err := PrepareWebScanTask(task, []byte(webTargetFixture), []byte(webSettingsFixture))
	if err != nil || prepared.Target == nil || prepared.Target.URL != "https://target.test/a?b=1" || prepared.RequestSet != nil {
		t.Fatalf("direct preparation: %v", err)
	}
	for _, change := range []struct{ source, settings string }{
		{webTargetFixture + " ", webSettingsFixture}, {webTargetFixture, webSettingsFixture + " "},
	} {
		if _, err := PrepareWebScanTask(task, []byte(change.source), []byte(change.settings)); err == nil {
			t.Fatal("changed exact input was accepted")
		}
	}
	task.Scan.TargetURL = "https://other.test/"
	if _, err := PrepareWebScanTask(task, []byte(webTargetFixture), []byte(webSettingsFixture)); err == nil {
		t.Fatal("changed assignment was accepted")
	}
}

func TestWebScanInventoryRejectsInvalidInputsBeforeTraffic(t *testing.T) {
	options := testInventoryOptions("scan")
	options.ApprovalRequirement = ApprovalNone
	for name, settings := range map[string][]byte{
		"missing":   []byte(`{"schema":"contractor.audit.web-scan-settings.v1","rateLimit":10}`),
		"null":      bytes.Replace([]byte(webSettingsFixture), []byte(`"tags":"http"`), []byte(`"tags":null`), 1),
		"alias":     bytes.Replace([]byte(webSettingsFixture), []byte(`"tags"`), []byte(`"Tags"`), 1),
		"duplicate": bytes.Replace([]byte(webSettingsFixture), []byte(`"rateLimit":7`), []byte(`"rateLimit":7,"rateLimit":8`), 1),
		"filter":    bytes.Replace([]byte(webSettingsFixture), []byte(`"http"`), []byte(`"http, bad"`), 1),
		"rate":      bytes.Replace([]byte(webSettingsFixture), []byte(`"rateLimit":7`), []byte(`"rateLimit":1001`), 1),
	} {
		t.Run(name, func(t *testing.T) {
			input := ExactInput{Name: "settings", Ref: options.SourceRef.Clone(), Digest: DigestBytes(settings)}
			input.Ref.Name = "settings"
			if _, err := BuildWebScanInventory([]byte(webTargetFixture), JSONMediaType, settings, input, options); err == nil {
				t.Fatal("invalid selection was accepted")
			}
		})
	}
}
