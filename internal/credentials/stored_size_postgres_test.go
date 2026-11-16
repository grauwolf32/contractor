package credentials

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Include escaping and numeric expansion; the real database is the size oracle.
func policyAtStoredSize(t *testing.T, ctx context.Context, pool *pgxpool.Pool, size int) EffectiveGatewayPolicy {
	t.Helper()
	budget := 1e100
	p := EffectiveGatewayPolicy{MaxBudget: &budget}
	for i := 0; i < 128; i++ {
		p.ModelPolicies = append(p.ModelPolicies, contracts.ModelPolicyRef{PolicyID: fmt.Sprintf("p%03d", i) + strings.Repeat("x", 124), Version: strings.Repeat("v", 64), Digest: "sha256:" + strings.Repeat("a", 64)})
	}
	measure := func() int {
		raw, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		var n int
		if err := pool.QueryRow(ctx, `SELECT octet_length($1::jsonb::text)`, raw).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	for width := 1; width < 240; width++ {
		p.Models = nil
		for i := 0; i < 128; i++ {
			p.Models = append(p.Models, fmt.Sprintf("m%03d", i)+strings.Repeat("x", width))
		}
		p.Models[127] += "é<&\"\\\n"
		if n := measure(); n >= size {
			p.Models[0] = p.Models[0][:len(p.Models[0])-(n-size)]
			if err := p.Validate(); err != nil {
				t.Fatal(err)
			}
			if measure() != size {
				t.Fatal("incorrect stored boundary")
			}
			return p
		}
	}
	t.Fatal("boundary fixture not found")
	return p
}

func TestCredentialStoredJSONBoundariesKeepTransactionUsable(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	for _, size := range []int{65536, 65537} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			record, _ := sealedTestRecord(t, fmt.Sprintf("size-%d", size), "sk-size-boundary")
			record.EffectivePolicy = policyAtStoredSize(t, ctx, pool, size)
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(ctx)
			repo := NewRepository(tx)
			var one int
			operation := Operation{OperationID: fmt.Sprintf("op-%d", size), IdempotencyKey: fmt.Sprintf("key-%d", size), RequestHash: "sha256:" + strings.Repeat("b", 64), CredentialID: record.CredentialID, Kind: OperationCreate, Phase: OperationPrepared, CreatedAt: record.CreatedAt, UpdatedAt: record.CreatedAt}
			operation.Request = []byte(`{"padding":"` + strings.Repeat("x", size-len(`{"padding": ""}`)) + `"}`)
			err = repo.InsertOperation(ctx, operation)
			if size == 65536 && err != nil || size > 65536 && !errors.Is(err, ErrInvalid) {
				t.Fatalf("journal write: %v", err)
			}
			if err := tx.QueryRow(ctx, `SELECT 1`).Scan(&one); err != nil {
				t.Fatalf("journal check aborted transaction: %v", err)
			}
			err = repo.InsertCredential(ctx, record)
			if size == 65536 && err != nil || size > 65536 && !errors.Is(err, ErrInvalid) {
				t.Fatalf("policy write: %v", err)
			}
			if err := tx.QueryRow(ctx, `SELECT 1`).Scan(&one); err != nil {
				t.Fatalf("policy check aborted transaction: %v", err)
			}
			if err := tx.Commit(ctx); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type boundedPolicyManager struct {
	*fakeGatewayManager
	preview, generated *EffectiveGatewayPolicy
}

func (m *boundedPolicyManager) ValidateCreate(ctx context.Context, r ManagerCreateRequest) (EffectiveGatewayPolicy, error) {
	p, err := m.fakeGatewayManager.ValidateCreate(ctx, r)
	if m.preview != nil {
		p = *m.preview
	}
	return p, err
}
func (m *boundedPolicyManager) Create(ctx context.Context, r ManagerCreateRequest) (GeneratedCredential, error) {
	g, err := m.fakeGatewayManager.Create(ctx, r)
	if m.generated != nil {
		g.EffectivePolicy = *m.generated
	}
	return g, err
}

func TestCredentialOversizedPolicyHasNoUnrecoverableIntent(t *testing.T) {
	for _, mode := range []string{"preflight", "generated", "cleanup unavailable", "recovery"} {
		t.Run(mode, func(t *testing.T) {
			pool, ctx := lifecycleTestPool(t)
			base := newFakeGatewayManager()
			f := newLifecycleFixture(t, pool, base, ServiceOptions{})
			manager := &boundedPolicyManager{fakeGatewayManager: base}
			oversized := policyAtStoredSize(t, ctx, pool, 65537)
			registry, err := NewManagerRegistry(ManagerRegistration{Implementation: llmgateway.LiteLLMVirtualKeysManager, Manager: manager})
			if err != nil {
				t.Fatal(err)
			}
			f.service.managers = registry
			if err := f.service.Recover(ctx); err != nil {
				t.Fatal(err)
			}
			request := f.createRequest("size-lifecycle", "size-lifecycle-key")
			switch mode {
			case "preflight":
				manager.preview = &oversized
			case "generated", "cleanup unavailable":
				manager.generated = &oversized
			case "recovery":
				f.service.afterCreate = func() error { return errors.New("crash after remote creation") }
			}
			if mode == "cleanup unavailable" {
				base.recoverErr = errors.New("remote cleanup unavailable")
			}
			_, err = f.service.Create(ctx, request)
			if err == nil {
				t.Fatal("oversized or interrupted create succeeded")
			}
			if mode == "preflight" {
				if !errors.Is(err, ErrInvalid) || base.createCalls() != 0 {
					t.Fatalf("preflight reached Gateway: %v", err)
				}
				if _, err = f.service.repository.GetOperationByIdempotency(ctx, OperationCreate, request.IdempotencyKey); !errors.Is(err, ErrNotFound) {
					t.Fatalf("preflight reserved an intent: %v", err)
				}
				return
			}
			if mode == "cleanup unavailable" || mode == "recovery" {
				op, err := f.service.repository.GetOperationByIdempotency(ctx, OperationCreate, request.IdempotencyKey)
				if err != nil || op.Phase != OperationPrepared || base.remoteCount() != 1 {
					t.Fatalf("uncertain cleanup must stay prepared: %s %v", op.Phase, err)
				}
			}
			if mode == "cleanup unavailable" {
				if err := f.service.Recover(ctx); !errors.Is(err, ErrGatewayUnavailable) {
					t.Fatalf("uncertain cleanup recovery: %v", err)
				}
				base.recoverErr = nil
			}
			if mode == "recovery" {
				manager.preview = &oversized
				f.service.afterCreate = nil
			}
			if err := f.service.Recover(ctx); err != nil {
				t.Fatalf("startup recovery: %v", err)
			}
			op, err := f.service.repository.GetOperationByIdempotency(ctx, OperationCreate, request.IdempotencyKey)
			if err != nil || op.Phase != OperationAbandoned || base.remoteCount() != 0 {
				t.Fatalf("permanent failure not settled: %s remote=%d err=%v", op.Phase, base.remoteCount(), err)
			}
			if _, err := f.service.Create(ctx, request); !errors.Is(err, ErrCreateAbandoned) {
				t.Fatalf("abandoned replay: %v", err)
			}
			if err := f.service.WithRunCreation(ctx, func() error { return nil }); err != nil {
				t.Fatalf("unrelated Run remains blocked: %v", err)
			}
		})
	}
}
