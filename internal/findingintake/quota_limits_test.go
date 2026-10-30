package findingintake

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The intake quotas are values of the limits registry: the receipts of one
// source Run (LIM-FND-46) and the evidence bytes of one proposal
// (LIM-FND-50). Each constant must equal its row, so neither can change alone.
func TestProposalQuotasAreTheirLimitsRegistryRows(t *testing.T) {
	registry, err := os.ReadFile("../../docs/spec/platform/limits.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, quota := range []struct {
		row   string
		value int64
		cell  string // the value cell's form, with the number as its first group
		unit  int64
	}{
		{row: "LIM-FND-46", value: maxProposalsPerRun, cell: `^([0-9,]+) receipts per source Run$`, unit: 1},
		{row: "LIM-FND-50", value: maxProposalEvidenceBytes, cell: `^([0-9,]+) MiB$`, unit: 1 << 20},
	} {
		if registered := registryValue(t, string(registry), quota.row, quota.cell) * quota.unit; registered != quota.value {
			t.Errorf("%s registers %d, but finding intake enforces %d", quota.row, registered, quota.value)
		}
	}
}

// registryValue returns the number in the value cell of one limits registry
// row, which must have the given form.
func registryValue(t *testing.T, registry, row, form string) int64 {
	t.Helper()
	rows := regexp.MustCompile("(?m)^\\| `"+regexp.QuoteMeta(row)+"` \\| ([^|]+) \\|").FindAllStringSubmatch(registry, -1)
	if len(rows) != 1 {
		t.Fatalf("limits.md has %d rows %s, want one", len(rows), row)
	}
	cell := strings.TrimSpace(rows[0][1])
	match := regexp.MustCompile(form).FindStringSubmatch(cell)
	if match == nil {
		t.Fatalf("%s value %q does not have the form %s", row, cell, form)
	}
	value, err := strconv.ParseInt(strings.ReplaceAll(match[1], ",", ""), 10, 64)
	if err != nil {
		t.Fatalf("%s value %q: %v", row, cell, err)
	}
	return value
}
