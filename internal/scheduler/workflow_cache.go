package scheduler

import (
	"container/list"
	"crypto/sha256"
	"sync"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// LIM-SCH-16 bounds the decoded Workflow snapshots one Scheduler keeps: two
// Runs per possible lane (LIM-OPS-7) and the stored snapshot text of every
// lane at the largest snapshot (LIM-CAT-4).
const (
	workflowSnapshotCacheRuns  = 2 * settingsstore.MaximumConcurrentRuns
	workflowSnapshotCacheBytes = settingsstore.MaximumConcurrentRuns * workflowconfig.MaxWorkflowSnapshotBytes
)

// workflowSnapshotCache keeps decoded and validated Workflow snapshots by Run
// (SCH-R167). A Run's snapshot is immutable, so a step whose Run still carries
// the same Workflow identity and snapshot digest reuses the validated value
// instead of decoding and validating it again. The cache owns its values and
// hands every caller a deep copy, so nothing one step changes reaches a later
// step. A snapshot that fails validation is never kept.
type workflowSnapshotCache struct {
	decode   func(runstore.WorkflowRun) (executableWorkflow, error)
	maxRuns  int
	maxBytes int

	mu      sync.Mutex
	entries map[string]*list.Element
	order   *list.List // most recently used first
	bytes   int
}

// workflowSnapshotKey is every Run field decodeExecutableWorkflow reads.
type workflowSnapshotKey struct {
	runID, schemaVersion, name, version string
	digest                              [sha256.Size]byte
}

type workflowSnapshotEntry struct {
	key      workflowSnapshotKey
	size     int
	workflow executableWorkflow
}

func newWorkflowSnapshotCache(
	decode func(runstore.WorkflowRun) (executableWorkflow, error), maxRuns, maxBytes int,
) *workflowSnapshotCache {
	return &workflowSnapshotCache{
		decode: decode, maxRuns: maxRuns, maxBytes: maxBytes,
		entries: make(map[string]*list.Element), order: list.New(),
	}
}

// executableWorkflow returns a caller-owned copy of run's decoded snapshot,
// decoding and validating it only when no entry matches run.
func (c *workflowSnapshotCache) executableWorkflow(run runstore.WorkflowRun) (executableWorkflow, error) {
	key := workflowSnapshotKey{
		runID: run.RunID, schemaVersion: run.WorkflowSchemaVersion,
		name: run.WorkflowName, version: run.WorkflowVersion,
		digest: sha256.Sum256(run.WorkflowSnapshot),
	}
	if cached, ok := c.lookup(key); ok {
		return cached.clone(), nil
	}
	decoded, err := c.decode(run)
	if err != nil {
		return executableWorkflow{}, err
	}
	c.store(key, len(run.WorkflowSnapshot), decoded)
	return decoded.clone(), nil
}

func (c *workflowSnapshotCache) lookup(key workflowSnapshotKey) (executableWorkflow, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	element, ok := c.entries[key.runID]
	if !ok || element.Value.(*workflowSnapshotEntry).key != key {
		return executableWorkflow{}, false
	}
	c.order.MoveToFront(element)
	// Entries are never mutated, so the copy is taken outside the lock.
	return element.Value.(*workflowSnapshotEntry).workflow, true
}

func (c *workflowSnapshotCache) store(key workflowSnapshotKey, size int, workflow executableWorkflow) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if element, ok := c.entries[key.runID]; ok {
		c.remove(element)
	}
	c.entries[key.runID] = c.order.PushFront(&workflowSnapshotEntry{key: key, size: size, workflow: workflow})
	c.bytes += size
	for c.order.Len() > c.maxRuns || c.bytes > c.maxBytes {
		c.remove(c.order.Back())
	}
}

func (c *workflowSnapshotCache) remove(element *list.Element) {
	entry := c.order.Remove(element).(*workflowSnapshotEntry)
	delete(c.entries, entry.key.runID)
	c.bytes -= entry.size
}

// clone returns a deep copy whose selected Stage belongs to the copy.
func (w executableWorkflow) clone() executableWorkflow {
	workflow := w.workflow.Clone()
	return executableWorkflow{workflow: workflow, stageName: w.stageName, stage: workflow.Stages[w.stageName]}
}
