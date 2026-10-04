package auditimport

import "testing"

func TestCWECatalogRejectsEmptyAndDuplicateIDs(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		document string
		want     string
	}{
		{
			document: `{"scheme":"CWE","version":"4.20","weakness_ids":["CWE-89",""]}`,
			want:     "bundled CWE catalog contains an empty ID",
		},
		{
			document: `{"scheme":"CWE","version":"4.20","weakness_ids":["CWE-89","CWE-89"]}`,
			want:     "bundled CWE catalog contains a duplicate ID",
		},
	} {
		if _, err := parseCWECatalog([]byte(test.document)); err == nil || err.Error() != test.want {
			t.Errorf("parseCWECatalog(%s) = %v, want %q", test.document, err, test.want)
		}
	}
}
