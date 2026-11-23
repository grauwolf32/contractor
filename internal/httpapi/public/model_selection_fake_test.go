package public

import (
	"context"
	"maps"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

type fakeModelSelection struct {
	mu       sync.Mutex
	value    settingsstore.ModelSelectionValue
	revision uint64
	updated  time.Time
}

func newFakeModelSelection() *fakeModelSelection {
	return &fakeModelSelection{value: settingsstore.ModelSelectionValue{DefaultModel: "fixture-a", Roles: map[string]string{}}, revision: 1, updated: time.Date(2026, 10, 21, 0, 0, 0, 0, time.UTC)}
}

func modelSelectionFixtureModels() []config.ModelChoice {
	models := make([]config.ModelChoice, 0, 2)
	for _, name := range []string{"fixture-a", "fixture-b"} {
		models = append(models, config.ModelChoice{ID: name, Digest: "sha256:" + strings.Repeat("a", 64), UpstreamModel: "test/" + name, ContextWindowTokens: 100000, MaxOutputTokens: 32768, Policies: map[string]string{
			"planner": "planner-" + name + "@1", "worker": "worker-" + name + "@1", "summarizer": "summary-" + name + "@1", "audit_completion_worker": "audit-" + name + "@1",
		}})
	}
	return models
}

func (f *fakeModelSelection) info() config.ModelSelectionInfo {
	value := f.value.Clone()
	updated := f.updated
	return config.ModelSelectionInfo{Configured: true, Models: modelSelectionFixtureModels(), Selection: &value, Revision: strconv.FormatUint(f.revision, 10), CatalogDigest: "sha256:" + strings.Repeat("b", 64), UpdatedAt: &updated}
}

func (f *fakeModelSelection) GetModelSelection(context.Context) (config.ModelSelectionInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.info(), nil
}

func (f *fakeModelSelection) ReplaceModelSelection(_ context.Context, value settingsstore.ModelSelectionValue, revision uint64, digest string) (config.ModelSelectionInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if value.Validate() != nil {
		return config.ModelSelectionInfo{}, settingsstore.ErrModelSelectionInvalid
	}
	known := func(id string) bool { return id == "fixture-a" || id == "fixture-b" }
	if !known(value.DefaultModel) {
		return config.ModelSelectionInfo{}, settingsstore.ErrModelSelectionInvalid
	}
	for _, id := range value.Roles {
		if !known(id) {
			return config.ModelSelectionInfo{}, settingsstore.ErrModelSelectionInvalid
		}
	}
	if revision != f.revision || digest != "sha256:"+strings.Repeat("b", 64) {
		return config.ModelSelectionInfo{}, settingsstore.ErrPrecondition
	}
	if value.DefaultModel != f.value.DefaultModel || !maps.Equal(value.Roles, f.value.Roles) {
		f.value = value.Clone()
		f.revision++
		f.updated = f.updated.Add(time.Microsecond)
	}
	return f.info(), nil
}
