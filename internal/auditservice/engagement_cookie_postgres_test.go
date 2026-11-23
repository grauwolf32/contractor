package auditservice

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/engagement"
)

func TestEngagementCookieAdmissionPinsHeldPrivateMaterial(t *testing.T) {
	f := newEngagementFixture(t, 1)
	material, err := credentials.NewTargetCookiesCredential(map[string]string{"session": "private-cookie-canary"})
	if err != nil {
		t.Fatal(err)
	}
	defer material.Destroy()
	if _, err = f.credentials.Create(f.ctx, credentials.RuntimeCredentialCreateRequest{CredentialID: "cookies-a", ActorID: f.project.OwnerID, IdempotencyKey: "cookies-a", Material: material}); err != nil {
		t.Fatal(err)
	}
	document := f.document("http://127.0.0.1")
	document.Identities = append(document.Identities, engagementcontract.Identity{ID: "account-a", Kind: "cookies", Origins: document.Identities[0].Origins, Credential: &runtimesettings.RuntimeCredentialRef{CredentialID: "cookies-a", Kind: runtimesettings.RuntimeCredentialTargetCookies}})
	draft, err := f.create(t, "cookies-pinned", document)
	if err != nil {
		t.Fatal(err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(started.Audit.BaselineSnapshot), "private-cookie-canary") {
		t.Fatal("baseline contains private cookie material")
	}
	projection := &engagementcontract.AllocationProjection{HTTP: &engagementcontract.HTTPAssignment{Identities: baseline.Engagement.Document.Identities}}
	private, err := engagement.MaterializeHTTPIdentities(f.ctx, f.credentials, projection)
	if err != nil || private["account-a"].Fields["session"].Reveal() != "private-cookie-canary" {
		t.Fatalf("private cookie consumer failed: %v", err)
	}
	if _, err = f.credentials.Delete(f.ctx, "cookies-a", f.project.OwnerID); !errors.Is(err, credentials.ErrRuntimeCredentialInUse) {
		t.Fatalf("cookie hold missing: %v", err)
	}
	decoded, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	clone := decoded.Engagement
	clone.Document.Identities[1].Credential.CredentialID = "other"
	if baseline.Engagement.Document.Identities[1].Credential.CredentialID != "cookies-a" {
		t.Fatal("identity clone retained mutable aliases")
	}
	clear(private["account-a"].Fields)
}
