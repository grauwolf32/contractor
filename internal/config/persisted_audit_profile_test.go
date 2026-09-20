package config

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestDecodeResolvedAuditProfileSnapshotRequiresCurrentRolesAndDigest(t *testing.T) {
	t.Parallel()
	root := copyAuditProfileConfigTree(t)
	writeAuditProfile(t, root, "legacy-profile", validAuditProfileYAML("legacy-profile"))
	profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("legacy-profile@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	// Frozen output of the removed digest algorithm for this fixture.
	const legacyDigest = "sha256:ca748004f9a6b65fc45fb7676e7c71a6f7ba67817dc8efb18f284a1815a29c39"
	setKind := func(value any) func(map[string]any) {
		return func(body map[string]any) {
			body["workflows"].(map[string]any)["check"].(map[string]any)["kind"] = value
		}
	}
	for _, test := range []struct {
		name      string
		mutate    func(map[string]any)
		wantError string
	}{
		{"current", func(map[string]any) {}, ""},
		{"legacy roles and digest", func(body map[string]any) {
			delete(body["workflows"].(map[string]any)["check"].(map[string]any), "kind")
			body["ref"].(map[string]any)["digest"] = legacyDigest
		}, "kind is invalid"},
		{"legacy digest with explicit roles", func(body map[string]any) {
			body["ref"].(map[string]any)["digest"] = legacyDigest
		}, "digest is invalid"},
		{"missing kind", func(body map[string]any) {
			delete(body["workflows"].(map[string]any)["check"].(map[string]any), "kind")
		}, "kind is invalid"},
		{"mixed roles", func(body map[string]any) {
			roles := body["workflows"].(map[string]any)
			other := make(map[string]any)
			for key, value := range roles["check"].(map[string]any) {
				other[key] = value
			}
			delete(other, "kind")
			roles["other"] = other
		}, "kind is invalid"},
		{"null kind", setKind(nil), "kind is invalid"},
		{"empty kind", setKind(""), "kind is invalid"},
		{"unknown kind", setKind("unknown"), "kind is invalid"},
		{"non-string kind", setKind(42), "cannot unmarshal"},
		{"changed valid kind", setKind("discovery"), "digest is invalid"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var body map[string]any
			if err := json.Unmarshal(encoded, &body); err != nil {
				t.Fatal(err)
			}
			test.mutate(body)
			raw, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			original := bytes.Clone(raw)
			decoded, err := DecodeResolvedAuditProfileSnapshot(raw)
			if test.wantError == "" {
				if err != nil || !reflect.DeepEqual(decoded, profile) {
					t.Fatalf("current profile changed during round trip: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), test.wantError) ||
				!reflect.DeepEqual(decoded, ResolvedAuditProfile{}) {
				t.Fatalf("decode = (%+v, %v), want empty profile and %q", decoded, err, test.wantError)
			}
			if !bytes.Equal(raw, original) {
				t.Fatal("decoder rewrote persisted snapshot bytes")
			}
		})
	}

	t.Run("all current role kinds", func(t *testing.T) {
		for _, kind := range []AuditWorkflowRoleKind{AuditWorkflowDiscovery, AuditWorkflowAssessment} {
			binding := profile.Workflows["check"]
			binding.Kind = kind
			profile.Workflows[string(kind)] = binding
		}
		profile.Ref.Digest, err = auditProfileDigest(Selector{ID: profile.Ref.Name, Version: profile.Ref.Version}, profile)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(profile)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeResolvedAuditProfileSnapshot(raw)
		if err != nil || !reflect.DeepEqual(decoded, profile) {
			t.Fatalf("current role kinds changed during round trip: %v", err)
		}
	})
}
