package postgres

import (
	"strings"
	"testing"
)

// TestNewAdvisoryKeyNeverJoinsTwoPartListsIntoOneKey pins the one property
// the advisory key builder exists for: two different namespaces and part
// lists never produce the same key, whatever separators or digits their
// values hold, and one list always produces the same key.
func TestNewAdvisoryKeyNeverJoinsTwoPartListsIntoOneKey(t *testing.T) {
	lists := [][]string{
		{"ns"}, {"ns", ""}, {"ns", "", ""}, {"ns", "a", "bc"}, {"ns", "ab", "c"}, {"ns", "abc"},
		{"ns", "a:b", "c"}, {"ns", "a", "b:c"}, {"ns", "1:a"}, {"ns", "", "1:a"}, {"ns1", ":a"},
		{"n", "s"}, {"ns", "a", "b", "c"}, {"ns", "ü", "b"}, {"ns", "üb"},
	}
	seen := map[AdvisoryKey]string{}
	for _, list := range lists {
		key := NewAdvisoryKey(list[0], list[1:]...)
		name := strings.Join(list, "|")
		if previous, exists := seen[key]; exists {
			t.Fatalf("parts %q and %q share advisory key %q", previous, name, key)
		}
		seen[key] = name
		if again := NewAdvisoryKey(list[0], list[1:]...); again != key {
			t.Fatalf("parts %q produced %q and then %q", name, key, again)
		}
	}
}

// TestReceiptKeysOfTheFamiliesNeverShareALock checks that the receipt keys of
// the three keyed mutation families serialize only requests under the same
// key: the same caller key in two families, two owners or two operations
// takes two locks.
func TestReceiptKeysOfTheFamiliesNeverShareALock(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	keys := map[string]ReceiptKey{
		"audit create":     AuditReceiptKey("owner", "audit.create", "key"),
		"audit start":      AuditReceiptKey("owner", "audit.start", "key"),
		"audit other":      AuditReceiptKey("other", "audit.create", "key"),
		"eval":             EvalReceiptKey("owner", "project", "resource", "audit.create", "key"),
		"eval resource":    EvalReceiptKey("owner", "project", "other", "audit.create", "key"),
		"runtime":          RuntimeManagementReceiptKey(digest),
		"runtime as audit": AuditReceiptKey("owner", "audit.create", digest),
	}
	seen := map[AdvisoryKey]string{}
	for name, key := range keys {
		lock := key.AdvisoryKey()
		if previous, exists := seen[lock]; exists {
			t.Fatalf("receipt keys %s and %s share advisory key %q", previous, name, lock)
		}
		seen[lock] = name
	}
}
