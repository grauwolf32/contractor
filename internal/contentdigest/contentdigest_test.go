package contentdigest

import "testing"

func TestDigestsArePinned(t *testing.T) {
	encoded, err := JSON(struct {
		AuditID  string `json:"auditId"`
		Revision uint64 `json:"revision"`
	}{"audit-1", 3})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, got, want string }{
		{"bytes", Bytes([]byte("hello")), "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"},
		{"empty", Bytes(nil), "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{"json", encoded, "sha256:aca7ac166a41c4b091572c8db65eaaf33cea54bdfc8f8814199d0d0a6dc99106"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s digest = %s, want %s", tc.name, tc.got, tc.want)
		}
	}
	if _, err := JSON(func() {}); err == nil {
		t.Fatal("JSON accepted an unencodable value")
	}
}
