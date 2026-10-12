package agentprincipal

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// RuntimeConfig label management reads the principals holding a label
// through LabelHolders: a rebind that would break an assigned principal's
// layer is refused and a label a principal holds cannot be deleted.
func TestPostgresLabelRebindCannotInvalidateAssignedPrincipalLayer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPrincipalPool(t, ctx)
	now := time.Date(2026, 9, 1, 2, 0, 0, 0, time.UTC)
	resolver := runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (llmgateway.ResolvedLLMGatewayConfig, error) {
		gatewayID, version, _ := strings.Cut(selector, "@")
		digit := "b"
		if gatewayID == "other-litellm" {
			digit = "c"
		}
		return llmgateway.ResolvedLLMGatewayConfig{
			Ref: llmgateway.LLMGatewayConfigRef{
				GatewayID: gatewayID, Version: version,
				Digest: "sha256:" + strings.Repeat(digit, 64),
			},
			Protocol: llmgateway.OpenAICompatibleProtocol,
			URL:      "http://127.0.0.1:4000/v1",
		}, nil
	})
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, GatewayResolver: resolver,
		RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name, gateway string) runtimeconfig.Ref {
		t.Helper()
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` +
			name + `","version":"1"},"spec":{"worker":{"llmGateway":{"gateway":"` + gateway + `@1"}}}}`)
		result, publishErr := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		return result.Version.Ref
	}
	firstRef := publish("route-one", "local-litellm")
	identicalRef := publish("route-two", "local-litellm")
	conflictingRef := publish("route-other", "other-litellm")
	repository := runtimeconfig.NewRepository(pool)
	for label, ref := range map[string]runtimeconfig.Ref{"route-one": firstRef, "route-two": identicalRef} {
		if _, err := repository.CreateBinding(ctx, label, ref, "operator", now); err != nil {
			t.Fatal(err)
		}
	}
	principalService, err := NewService(ServiceOptions{
		Pool: pool, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principalID := strings.Repeat("d", 64)
	if _, err := principalService.Register(
		ctx, principalID, []string{"route-one", "route-two"},
	); err != nil {
		t.Fatal(err)
	}
	bindingService, err := runtimeconfig.NewBindingService(allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := runtimeconfig.NewManagementService(pool, publisher, bindingService, LabelHolders{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := management.Rebind(
		ctx, "route-two", 1, conflictingRef, "rebind-conflict", "operator", now.Add(time.Minute),
	); err == nil {
		t.Fatal("conflicting label rebind was accepted")
	} else {
		var conflict *runtimeconfig.MergeConflictError
		if !errors.As(err, &conflict) || conflict.Path != "worker.llmGateway.gateway" {
			t.Fatalf("conflicting rebind error = %v", err)
		}
	}
	current, err := repository.GetBinding(ctx, "route-two")
	if err != nil || current.Revision != 1 || current.Ref != identicalRef {
		t.Fatalf("binding after rolled-back conflict = (%+v, %v)", current, err)
	}
	_, err = management.DeleteBinding(ctx, "route-one", 1, "delete-assigned", "operator", now.Add(2*time.Minute))
	var inUse *runtimeconfig.LabelInUseError
	if !errors.Is(err, runtimeconfig.ErrConflict) || !errors.As(err, &inUse) ||
		len(inUse.RuntimeAgentIDs) != 1 || inUse.RuntimeAgentIDs[0] != principalID {
		t.Fatalf("delete assigned label error = %v", err)
	}
}

// TestPostgresRebindUnrelatedLabelRemovalIsPrecondition proves that when a
// rebind optimistically lists a principal holding another label and that other
// label's binding is removed before the lock acquires it, the rebind reports a
// precondition failure (412) rather than a raw not-found that the public
// handler would answer as 400 runtime_label_unknown for a label the request
// never selected.
func TestPostgresRebindUnrelatedLabelRemovalIsPrecondition(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPrincipalPool(t, ctx)
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	refs := map[string]runtimeconfig.Ref{}
	repository := runtimeconfig.NewRepository(pool)
	for _, name := range []string{"selected", "unrelated"} {
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name + `","version":"1"},"spec":{"worker":{"telemetry":null}}}`)
		published, publishErr := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		refs[name] = published.Version.Ref
		if _, err := repository.CreateBinding(ctx, name, published.Version.Ref, "operator", now); err != nil {
			t.Fatal(err)
		}
	}

	principalService, err := NewService(ServiceOptions{
		Pool: pool, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	// One principal holds both labels, so a rebind of "selected" also locks
	// "unrelated".
	if _, err := principalService.Register(
		ctx, strings.Repeat("e", 64), []string{"selected", "unrelated"},
	); err != nil {
		t.Fatal(err)
	}

	bindings, err := runtimeconfig.NewBindingService(allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := runtimeconfig.NewManagementService(pool, publisher, bindings, LabelHolders{})
	if err != nil {
		t.Fatal(err)
	}

	// Remove the unrelated label's binding behind the optimistic snapshot the
	// rebind already read: the principal row still references "unrelated", so
	// the lock set still includes it, but its binding row is gone.
	if _, err := pool.Exec(ctx, `DELETE FROM runtime_label_bindings WHERE label = 'unrelated'`); err != nil {
		t.Fatal(err)
	}

	_, err = management.Rebind(ctx, "selected", 1, refs["selected"], "rebind-race", "operator", now.Add(time.Minute))
	if !errors.Is(err, runtimeconfig.ErrPrecondition) {
		t.Fatalf("concurrent unrelated-label removal rebind error = %v, want ErrPrecondition", err)
	}
	if errors.Is(err, runtimeconfig.ErrNotFound) || errors.Is(err, runtimeconfig.ErrLabelNotFound) {
		t.Fatalf("rebind reported the selected label as unknown: %v", err)
	}
	// The selected label's binding is untouched.
	if current, getErr := repository.GetBinding(ctx, "selected"); getErr != nil || current.Revision != 1 {
		t.Fatalf("selected binding after failed rebind = (%+v, %v)", current, getErr)
	}
}
