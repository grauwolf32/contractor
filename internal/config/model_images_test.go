package config

import (
	"path/filepath"
	"testing"
)

func TestManagedImageSupportIsPublishedPinnedAndRecovered(t *testing.T) {
	for _, supported := range []bool{false, true} {
		t.Run(map[bool]string{false: "text-only", true: "vision"}[supported], func(t *testing.T) {
			operator, managed := copyCoreFixture(t), filepath.Join(t.TempDir(), "managed")
			manager := newTestManager(t, operator, managed, ManagerOptions{})
			request := validPolicyPublication("publish-image-policy")
			request.ModelPolicy.SupportsImages = &supported
			result, err := manager.Publish(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			if result.Resource.Body.(map[string]any)["supportsImages"] != supported {
				t.Fatal("publication omitted explicit support metadata")
			}
			recovered := newTestManager(t, operator, managed, ManagerOptions{})
			policy, err := recovered.ModelPolicy(request.Name + "@" + request.Version)
			if err != nil || policy.SupportsImages == nil || *policy.SupportsImages != supported {
				t.Fatalf("recovered metadata: %+v %v", policy, err)
			}
			if policy.Model != request.ModelPolicy.Model {
				t.Fatal("image support changed the configured model")
			}
			opposite := !supported
			request.ModelPolicy.SupportsImages = &opposite
			request.IdempotencyKey = "changed-images"
			if _, err := recovered.Publish(t.Context(), request); err == nil {
				t.Fatal("changed image support reused an immutable policy identity")
			}
		})
	}
}
