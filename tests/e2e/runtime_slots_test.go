//go:build e2e

package e2e

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
)

func TestRuntimeAllocationSlotsAcrossProductionProcesses(t *testing.T) {
	if os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is unset")
	}
	h := startPlacementStackWithConfiguration(t, func(root string) {
		path := filepath.Join(root, "agent-templates", "probe.yaml")
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		body := strings.ReplaceAll(string(raw), "value: 10}", "value: 120}")
		body = strings.ReplaceAll(body, "timeoutSeconds: 20", "timeoutSeconds: 180")
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	})
	_, etag := getSchedulerSettings(t, h.operations.client, h.operations.baseURL)
	h.operations.request(http.MethodPut, "/v1/operations/settings/scheduler", map[string]int{"maxConcurrentRuns": 1, "maxConcurrentToolRuns": 5}, http.StatusOK, map[string]string{"If-Match": etag})
	markers := filepath.Join(h.temporary, "scan-started")
	release := filepath.Join(h.temporary, "scan-release")
	if err := os.Mkdir(markers, 0700); err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf(`import json,os,time
from pathlib import Path
Path(%q,str(os.getpid())).write_text("started")
while not Path(%q).exists(): time.sleep(0.02)
print(json.dumps({"template-id":"slots-fixture","info":{"severity":"info"}}))
`, markers, release)
	if err := os.WriteFile(filepath.Join(h.scannerBin, "nuclei.py"), []byte(script), 0600); err != nil {
		t.Fatal(err)
	}
	h.startRuntimeSlots("runtime-four", "local/slots", true, 4)
	snapshot := func() ([]controlplane.RuntimeAgentObservation, []controlplane.AllocationObservation) {
		var result struct {
			RuntimeAgents []controlplane.RuntimeAgentObservation `json:"runtimeAgents"`
			Allocations   []controlplane.AllocationObservation   `json:"allocations"`
		}
		h.decode(t, h.operations.request(http.MethodGet, "/v1/operations/snapshot", nil, http.StatusOK, nil), &result)
		return result.RuntimeAgents, result.Allocations
	}
	wait := func(ready func() bool) {
		t.Helper()
		for !ready() {
			select {
			case <-h.ctx.Done():
				t.Fatal("Runtime slots scenario timed out")
			case <-time.After(20 * time.Millisecond):
			}
		}
	}
	runs := make([]string, 5)
	for index := range 4 {
		runs[index] = h.create(t, map[string]string{"local": "slots"}, "")
	}
	wait(func() bool { rows, err := os.ReadDir(markers); return err == nil && len(rows) == 4 })
	agents, allocations := snapshot()
	if len(agents) != 1 || agents[0].AllocationSlots != 4 || len(allocations) != 4 {
		t.Fatalf("four allocations did not share Runtime: %+v %+v", agents, allocations)
	}
	instance := agents[0].InstanceID
	for _, allocation := range allocations {
		if allocation.RuntimeAgentInstanceID != instance {
			t.Fatal("allocation escaped shared instance")
		}
	}
	runs[4] = h.create(t, map[string]string{"local": "slots"}, "")
	waitForUnallocatedPreparingExecution(t, h.ctx, h.store, runs[4])
	cancelWorkflowRun(t, h.operations.client, h.operations.baseURL, runs[0])
	h.terminal(t, runs[0], "cancelled")
	wait(func() bool {
		_, items := snapshot()
		if len(items) != 4 {
			return false
		}
		ids := make([]string, 0, 4)
		for _, item := range items {
			if item.RuntimeAgentInstanceID != instance {
				return false
			}
			ids = append(ids, item.RunID)
		}
		for _, id := range runs[1:] {
			if !slices.Contains(ids, id) {
				return false
			}
		}
		return true
	})
	if err := os.WriteFile(release, []byte("continue"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, id := range runs[1:] {
		h.complete(t, id, "local/slots")
	}
	wait(func() bool {
		agents, items := snapshot()
		return len(items) == 0 && len(agents) == 1 && agents[0].AvailableSlots == 4
	})
	h.stopRuntime("runtime-four")
	h.startRuntimeSlots("runtime-four", "local/slots", true, 1)
	agents, _ = snapshot()
	if len(agents) != 1 || agents[0].InstanceID == instance || agents[0].AllocationSlots != 1 {
		t.Fatalf("lower-capacity restart: %+v", agents)
	}
	h.complete(t, h.create(t, map[string]string{"local": "slots"}, ""), "local/slots")
}
