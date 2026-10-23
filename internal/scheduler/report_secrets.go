package scheduler

import (
	"container/list"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"sync"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// LIM-SCH-19 bounds the Stages whose preparation secrets one Scheduler keeps
// for report redaction: two per possible lane of both pools (LIM-OPS-7,
// LIM-OPS-34), as for decoded Workflow snapshots (LIM-SCH-16).
const preparedSecretStages = 2 * settingsstore.MaximumLanes

// preparedSecretCache keeps, by StageExecution and allocation, the secret
// values materialized for the Stage's Worker preparation, so that report
// persistence of the same Stage redacts with exactly those values instead of
// resolving and decrypting the credentials again (SCH-R178). The values stay
// self-redacting contracts.SecretString in process memory (SEC-R88); nothing
// durable, logged or exported carries them. A Stage has an entry from its
// latest successful materialization until its terminal release begins, and
// the least recently stored Stage is evicted beyond LIM-SCH-19. A Stage
// without an entry, after a restart or an eviction, resolves its pinned
// credentials again.
type preparedSecretCache struct {
	maxStages int

	mu      sync.Mutex
	entries map[string]*list.Element
	order   *list.List // most recently stored first
}

type preparedSecretEntry struct {
	stageExecutionID string
	allocations      map[string][]contracts.SecretString
}

// The cache and its entries never format, encode or log the kept values:
// fmt cannot reach SecretString's own redaction through their unexported
// fields, so each type redacts itself as a whole for every verb, for
// encoding/json and for log/slog (SEC-R88). The entry uses value receivers so
// that a copied entry redacts too; the cache, which holds a mutex, is only
// used through its pointer.

func (e preparedSecretEntry) String() string {
	return fmt.Sprintf("preparedSecretEntry(%q, %d allocations, [REDACTED])", e.stageExecutionID, len(e.allocations))
}

func (e preparedSecretEntry) GoString() string { return e.String() }

func (e preparedSecretEntry) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, e.String())
}

func (e preparedSecretEntry) MarshalJSON() ([]byte, error) { return json.Marshal(e.String()) }

func (e preparedSecretEntry) LogValue() slog.Value { return slog.StringValue(e.String()) }

func (c *preparedSecretCache) String() string {
	if c == nil {
		return "preparedSecretCache(nil)"
	}
	return fmt.Sprintf("preparedSecretCache(at most %d Stages, [REDACTED])", c.maxStages)
}

func (c *preparedSecretCache) GoString() string { return c.String() }

func (c *preparedSecretCache) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, c.String())
}

func (c *preparedSecretCache) MarshalJSON() ([]byte, error) { return json.Marshal(c.String()) }

func (c *preparedSecretCache) LogValue() slog.Value { return slog.StringValue(c.String()) }

func newPreparedSecretCache(maxStages int) *preparedSecretCache {
	return &preparedSecretCache{maxStages: maxStages, entries: make(map[string]*list.Element), order: list.New()}
}

// store replaces the Stage's entry with the secret values of settings, keyed
// by the allocation of each reserved logical Agent that has settings.
func (c *preparedSecretCache) store(
	stageExecutionID string,
	reservations []controlplane.Reservation,
	settings map[string]runtimesettings.WorkerExecutionSettings,
) {
	if c == nil {
		return
	}
	allocations := make(map[string][]contracts.SecretString, len(reservations))
	for _, reservation := range reservations {
		value, ok := settings[reservation.Grant.LogicalAgentName]
		if !ok {
			continue
		}
		revealed := value.RuntimeSettings.SecretValues()
		kept := make([]contracts.SecretString, len(revealed))
		for index, secret := range revealed {
			kept[index] = contracts.NewSecretString(secret)
		}
		allocations[reservation.Grant.AllocationID] = kept
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if element, ok := c.entries[stageExecutionID]; ok {
		c.remove(element)
	}
	c.entries[stageExecutionID] = c.order.PushFront(&preparedSecretEntry{
		stageExecutionID: stageExecutionID, allocations: allocations,
	})
	for c.order.Len() > c.maxStages {
		c.remove(c.order.Back())
	}
}

// lookup returns a caller-owned copy of the values kept for one allocation of
// the Stage, and whether the Stage's preparation kept any for it.
func (c *preparedSecretCache) lookup(stageExecutionID, allocationID string) ([]string, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	element, ok := c.entries[stageExecutionID]
	if !ok {
		return nil, false
	}
	kept, ok := element.Value.(*preparedSecretEntry).allocations[allocationID]
	if !ok {
		return nil, false
	}
	values := make([]string, len(kept))
	for index, secret := range kept {
		values[index] = secret.Reveal()
	}
	return values, true
}

// forget drops the Stage's entry once its lifecycle in this process ends.
func (c *preparedSecretCache) forget(stageExecutionID string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if element, ok := c.entries[stageExecutionID]; ok {
		c.remove(element)
	}
}

func (c *preparedSecretCache) remove(element *list.Element) {
	entry := c.order.Remove(element).(*preparedSecretEntry)
	delete(c.entries, entry.stageExecutionID)
}
