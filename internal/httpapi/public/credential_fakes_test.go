package public

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

type fakeManagedCredentials struct {
	mu        sync.Mutex
	records   map[string]credentials.Record
	lookups   map[string]config.CredentialMetadata
	created   map[string]string
	deleted   map[string]string
	guarded   int
	guard     func(func() error) error
	deleteErr error
}

func newFakeManagedCredentials() *fakeManagedCredentials {
	return &fakeManagedCredentials{
		records: make(map[string]credentials.Record), lookups: make(map[string]config.CredentialMetadata),
		created: make(map[string]string), deleted: make(map[string]string),
	}
}

func (f *fakeManagedCredentials) ListCredentials(
	_ context.Context, after string, limit int,
) ([]credentials.Record, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	ids := make([]string, 0, len(f.records))
	for id := range f.records {
		if id > after {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	if len(ids) > limit {
		ids = ids[:limit]
	}
	result := make([]credentials.Record, 0, len(ids))
	for _, id := range ids {
		result = append(result, f.records[id])
	}
	return result, nil
}

func (f *fakeManagedCredentials) GetCredential(_ context.Context, id string) (credentials.Record, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	record, exists := f.records[id]
	if !exists {
		return credentials.Record{}, credentials.ErrNotFound
	}
	return record, nil
}

func (f *fakeManagedCredentials) Create(
	_ context.Context, request credentials.CreateRequest,
) (credentials.CreateResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if previous, exists := f.created[request.IdempotencyKey]; exists {
		if previous != request.CredentialID {
			return credentials.CreateResult{}, credentials.ErrConflict
		}
		return credentials.CreateResult{Credential: f.records[previous], Replayed: true}, nil
	}
	if _, exists := f.records[request.CredentialID]; exists {
		return credentials.CreateResult{}, credentials.ErrConflict
	}
	refs := append([]contracts.ModelPolicyRef(nil), request.GatewayPolicy.ModelPolicies...)
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].PolicyID != refs[j].PolicyID {
			return refs[i].PolicyID < refs[j].PolicyID
		}
		if refs[i].Version != refs[j].Version {
			return refs[i].Version < refs[j].Version
		}
		return refs[i].Digest < refs[j].Digest
	})
	record := credentials.Record{
		CredentialID: request.CredentialID, LLMGateway: request.LLMGateway, Label: request.Label,
		EffectivePolicy: credentials.EffectiveGatewayPolicy{
			ModelPolicies: refs, Models: []string{"test-model"},
			MaxBudget: request.GatewayPolicy.MaxBudget, BudgetDuration: request.GatewayPolicy.BudgetDuration,
			TPMLimit: request.GatewayPolicy.TPMLimit, RPMLimit: request.GatewayPolicy.RPMLimit,
			MaxParallelRequests: request.GatewayPolicy.MaxParallelRequests,
		},
		CreatedAt: time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
	}
	f.records[request.CredentialID] = record
	f.created[request.IdempotencyKey] = request.CredentialID
	return credentials.CreateResult{Credential: record}, nil
}

func (f *fakeManagedCredentials) Delete(
	_ context.Context, request credentials.DeleteRequest,
) (credentials.DeleteResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleteErr != nil {
		return credentials.DeleteResult{}, f.deleteErr
	}
	if previous, exists := f.deleted[request.IdempotencyKey]; exists {
		if previous != request.CredentialID {
			return credentials.DeleteResult{}, credentials.ErrConflict
		}
		return credentials.DeleteResult{Replayed: true}, nil
	}
	if _, exists := f.records[request.CredentialID]; !exists {
		return credentials.DeleteResult{}, credentials.ErrNotFound
	}
	delete(f.records, request.CredentialID)
	f.deleted[request.IdempotencyKey] = request.CredentialID
	return credentials.DeleteResult{}, nil
}

func (f *fakeManagedCredentials) WithRunCreation(_ context.Context, fn func() error) error {
	f.mu.Lock()
	f.guarded++
	guard := f.guard
	f.mu.Unlock()
	if guard != nil {
		return guard(fn)
	}
	return fn()
}

func (f *fakeManagedCredentials) LookupLLMCredential(
	ctx context.Context, id string,
) (config.CredentialMetadata, error) {
	record, err := f.GetCredential(ctx, id)
	if err == nil {
		return config.CredentialMetadata{
			Ref: contracts.LLMCredentialRef{CredentialID: id}, LLMGateway: record.LLMGateway,
		}, nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	metadata, ok := f.lookups[id]
	if !ok {
		return config.CredentialMetadata{}, err
	}
	return metadata, nil
}

var _ ManagedCredentialLifecycle = (*fakeManagedCredentials)(nil)
var _ config.CredentialLookup = (*fakeManagedCredentials)(nil)

func stringPointer(value string) *string { return &value }
