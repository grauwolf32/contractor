package auditservice

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

type privateMaterialFixture func(context.Context, string, []runtimesettings.RuntimeCredentialKind, func(runtimesettings.RuntimeCredentialKind, []byte) error) error

func (f privateMaterialFixture) UsePlaintext(ctx context.Context, id string, kinds []runtimesettings.RuntimeCredentialKind, consume func(runtimesettings.RuntimeCredentialKind, []byte) error) error {
	return f(ctx, id, kinds, consume)
}

func TestEngagementMaterialUsesOnlyExactAssignedReferences(t *testing.T) {
	ref := runtimesettings.RuntimeCredentialRef{CredentialID: "held-a", Kind: runtimesettings.RuntimeCredentialTargetCookies}
	projection := &engagementcontract.AllocationProjection{HTTP: &engagementcontract.HTTPAssignment{Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous"}, {ID: "account-a", Kind: "cookies", Credential: &ref}}}}
	calls := 0
	resolver := privateMaterialFixture(func(_ context.Context, id string, kinds []runtimesettings.RuntimeCredentialKind, consume func(runtimesettings.RuntimeCredentialKind, []byte) error) error {
		calls++
		if id != "held-a" || len(kinds) != 1 || kinds[0] != ref.Kind {
			t.Fatal("unassigned credential lookup")
		}
		return consume(ref.Kind, []byte(`{"cookies":{"session":"private-canary"}}`))
	})
	material, err := MaterializeHTTPIdentities(context.Background(), resolver, projection)
	if err != nil || calls != 1 || len(material) != 1 || material["account-a"].Fields["session"].Reveal() != "private-canary" {
		t.Fatalf("unexpected material result: calls=%d err=%v", calls, err)
	}

}

func TestEngagementMaterialFailureIsRedactedAndCancellationRetained(t *testing.T) {
	ref := runtimesettings.RuntimeCredentialRef{CredentialID: "held-a", Kind: runtimesettings.RuntimeCredentialTargetBearer}
	projection := &engagementcontract.AllocationProjection{HTTP: &engagementcontract.HTTPAssignment{Identities: []engagementcontract.Identity{{ID: "account-a", Credential: &ref}}}}
	resolver := privateMaterialFixture(func(context.Context, string, []runtimesettings.RuntimeCredentialKind, func(runtimesettings.RuntimeCredentialKind, []byte) error) error {
		return errors.New("private-canary")
	})
	result, err := MaterializeHTTPIdentities(context.Background(), resolver, projection)
	if result != nil || err == nil || strings.Contains(err.Error(), "private-canary") {
		t.Fatal("private provider error leaked")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = MaterializeHTTPIdentities(ctx, resolver, projection); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation was lost")
	}
}
