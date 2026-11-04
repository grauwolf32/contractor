package auditservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// TestRequestDigestsArePinned pins the canonical encoding of each Audit
// request digest, so a change to what an idempotency key binds is deliberate
// rather than a side effect of a refactoring: stored receipts compare these
// values. The values predate the move out of the public handlers.
func TestRequestDigestsArePinned(t *testing.T) {
	t.Parallel()

	revision := "rev-1"
	for _, tc := range []struct {
		name string
		got  func() (string, error)
		want string
	}{
		{"audit create", func() (string, error) {
			return CreateDraftRequestDigest(CreateDraftParams{ProjectID: "project-1", Profile: ProfileSelector{Name: "profile"}})
		}, "sha256:6cbc66c38782dec0a016581539d2fa6602c3ecdff50d6ac91291f98dbe5015ba"},
		{"audit create selection", func() (string, error) {
			return CreateDraftRequestDigest(CreateDraftParams{
				Labels: map[string]string{"team": "appsec", "cloud": "aws"}, ProjectID: "project-1",
				Profile: ProfileSelector{Name: "profile", Version: "1.0.0"},
				Inputs: map[string]contracts.ArtifactRef{
					"source": {Namespace: "project-1", Name: "repo", Revision: &revision},
				},
				RuntimeLabels: []string{"zone-b", "zone-a"}, Scope: Scope{Objective: "o", Target: "t"},
				AuditID: "audit-1", OwnerID: "user-1", IdempotencyKey: "key-1",
			})
		}, "sha256:6defea2f8302e0d2d0e02a3568a327faf12d698a4406b6eda320d20445ac8e4a"},
		{"audit start", lifecycle("start", nil), "sha256:aaa21d0388ca0a5e7fc3b9b66ba8cd9f93ba2847c42950363e601b547bdf54be"},
		{"audit start limit", lifecycle("start", timeLimit(0)), "sha256:c9864979e1a2a956eb06cf99e96075ed3eea75d72f9e3a5ad740f4713fab91be"},
		{"audit pause", lifecycle("pause", nil), "sha256:d651d1f2a09390f72cb8e3bb6847ded96ba65d4c561fe5922a19811bb8e4219a"},
		{"audit resume", lifecycle("resume", nil), "sha256:3996166e91fb7c0637e356be5a80795089ca89650a32cb779ad9b812a38bb7ea"},
		{"audit resume limit", lifecycle("resume", timeLimit(3600)), "sha256:55de97d14069029d34332b1642373072a616e96c0b091f990effb512f881c443"},
		{"audit cancel", lifecycle("cancel", nil), "sha256:80337beddf6f790375e0f8197a444e53d9c285f4c9b26332b2eae863e1098c0d"},
		{"audit delete", lifecycle("delete", nil), "sha256:4f5fc567729e7acbdca9c3a8067ceec86d32044a4aad7bf57dbe1a3a03d42afb"},
	} {
		got, err := tc.got()
		if err != nil || got != tc.want {
			t.Errorf("%s digest = %s (%v), want %s", tc.name, got, err, tc.want)
		}
	}
}

// Every Audit lifecycle request binds its action, a start included: no action
// is encoded by omission, and a time limit is part of the request.
func TestLifecycleRequestDigestBindsEveryActionAndLimit(t *testing.T) {
	t.Parallel()

	seen := map[string]string{}
	for _, action := range []string{"start", "pause", "resume", "cancel", "delete"} {
		digest, err := LifecycleRequestDigest(action, "audit-1", 3, nil)
		want := contentdigest.Bytes([]byte(`{"action":"` + action + `","auditId":"audit-1","revision":3}`))
		if err != nil || digest != want {
			t.Errorf("%s digest = %s (%v), want the digest of its canonical request %s", action, digest, err, want)
		}
		if other, found := seen[digest]; found {
			t.Errorf("%s and %s share a request digest", action, other)
		}
		seen[digest] = action
	}
	unlimited, _ := LifecycleRequestDigest("start", "audit", 1, nil)
	if limited, _ := LifecycleRequestDigest("start", "audit", 1, timeLimit(0)); unlimited == limited {
		t.Fatal("unlimited start shares default request digest")
	}
	hour, _ := LifecycleRequestDigest("resume", "audit", 1, timeLimit(3600))
	if none, _ := LifecycleRequestDigest("resume", "audit", 1, timeLimit(0)); hour == none {
		t.Fatal("resume limits share request digest")
	}
}

// TestCreateDraftRequestDigestUsesCanonicalSelection binds the normalized
// metadata and Runtime labels, so an equivalent selection replays and an
// invalid one is rejected before any store call.
func TestCreateDraftRequestDigestUsesCanonicalSelection(t *testing.T) {
	t.Parallel()

	params := CreateDraftParams{ProjectID: "project", Labels: map[string]string{"cloud": "aws", "team": "appsec"}}
	first, err := CreateDraftRequestDigest(params)
	if err != nil {
		t.Fatal(err)
	}
	params.Labels = map[string]string{"team": "appsec", "cloud": "aws"}
	if reordered, err := CreateDraftRequestDigest(params); err != nil || reordered != first {
		t.Fatalf("equivalent selection changed digest: %v", err)
	}
	params.Labels = map[string]string{"local": ""}
	if changed, err := CreateDraftRequestDigest(params); err != nil || changed == first {
		t.Fatalf("different selection reused digest: %v", err)
	}
	params.Labels = nil
	omitted, err := CreateDraftRequestDigest(params)
	if err != nil {
		t.Fatal(err)
	}
	params.Labels = map[string]string{}
	if empty, err := CreateDraftRequestDigest(params); err != nil || empty != omitted {
		t.Fatalf("empty selection differs from omission: %v", err)
	}
	params.RuntimeLabels = []string{"zone-b", "zone-a"}
	sorted, err := CreateDraftRequestDigest(params)
	if err != nil {
		t.Fatal(err)
	}
	params.RuntimeLabels = []string{"zone-a", "zone-b"}
	if reordered, err := CreateDraftRequestDigest(params); err != nil || reordered != sorted {
		t.Fatalf("reordered Runtime labels changed digest: %v", err)
	}

	params.Labels = map[string]string{"cloud": "", "local": ""}
	if _, err := CreateDraftRequestDigest(params); !errors.Is(err, ErrInvalid) {
		t.Fatalf("mixed namespaces error = %v, want ErrInvalid", err)
	}
	params.Labels, params.RuntimeLabels = nil, []string{"zone-a", "zone-a"}
	if _, err := CreateDraftRequestDigest(params); !errors.Is(err, runtimeconfig.ErrInvalid) {
		t.Fatalf("duplicate Runtime labels error = %v, want the RuntimeConfig error", err)
	}
}

func lifecycle(action string, seconds *int) func() (string, error) {
	return func() (string, error) { return LifecycleRequestDigest(action, "audit-1", 3, seconds) }
}

func timeLimit(value int) *int { return &value }
