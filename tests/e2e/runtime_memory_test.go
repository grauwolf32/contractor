//go:build e2e

package e2e

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
)

// The opt-in driver samples these real registered process trees. Normal release
// execution keeps the deterministic four-active/fifth-waits scenario unchanged.
func measureRuntimeAllocationSlots(t *testing.T, directory string) {
	const rounds = 1 << 20
	slots, err := strconv.Atoi(os.Getenv("CONTRACTOR_RUNTIME_MEMORY_SLOTS"))
	image, target := os.Getenv("CONTRACTOR_TEST_PODMAN_IMAGE"), os.Getenv("CONTRACTOR_RUNTIME_MEMORY_TARGET")
	if runtime.GOOS != "linux" || (slots != 1 && slots != 4) || err != nil ||
		os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" || image == "" || target == "" || !filepath.IsAbs(directory) {
		t.Fatal("memory measurement requires Linux, PostgreSQL, a pinned Podman image, target and an absolute output directory; slots must be 1 or 4")
	}
	h := startPlacementStackWithConfiguration(t, func(root string) {
		path := filepath.Join(root, "agent-templates", "probe.yaml")
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		value := strings.ReplaceAll(string(body), "value: 10}", "value: 120}")
		value = strings.ReplaceAll(value, "timeoutSeconds: 20", "timeoutSeconds: 180")
		if err := os.WriteFile(path, []byte(value), 0600); err != nil {
			t.Fatal(err)
		}
	})
	ctx, stop := signal.NotifyContext(h.ctx, os.Interrupt)
	h.ctx = ctx
	t.Cleanup(stop)
	h.runtimePath = os.Getenv("PATH")
	record := func(name string, value any) {
		t.Helper()
		body, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(directory, name+".json")
		if err := os.WriteFile(path+".tmp", body, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(path+".tmp", path); err != nil {
			t.Fatal(err)
		}
	}
	wait := func(ready func() bool) {
		t.Helper()
		for !ready() {
			select {
			case <-h.ctx.Done():
				t.Fatal("memory measurement barrier cancelled or expired")
			case <-time.After(20 * time.Millisecond):
			}
		}
	}
	workers := filepath.Join(directory, "workers")
	if err := os.Mkdir(workers, 0700); err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf(`import hashlib,json,os,sys,time,urllib.request
from pathlib import Path
root=Path(%q)
def record(kind,value):
 path=root/(kind+"-"+str(os.getpid())+".json")
 temporary=path.with_suffix(".tmp")
 temporary.write_text(json.dumps(value))
 temporary.replace(path)
def wait(name):
 while not (root.parent/name).exists(): time.sleep(0.02)
with urllib.request.urlopen(sys.argv[sys.argv.index("-u")+1],timeout=10) as response:
 body=response.read(65537)
if len(body)!=65536: raise RuntimeError("unexpected measurement input")
for _ in range(64): hashlib.sha256(body).digest()
record("ready",{"pid":os.getpid(),"input_sha256":hashlib.sha256(body).hexdigest()})
wait("load")
start=time.monotonic()
for _ in range(%d): digest=hashlib.sha256(body).hexdigest()
record("result",{"start":start,"end":time.monotonic(),"units":%d,"digest":digest})
wait("release")
print(json.dumps({"template-id":"memory-fixture","info":{"severity":"info"}}))
`, workers, rounds, rounds)
	if err := os.WriteFile(filepath.Join(h.scannerBin, "nuclei.py"), []byte(script), 0600); err != nil {
		t.Fatal(err)
	}
	_, etag := getSchedulerSettings(t, h.operations.client, h.operations.baseURL)
	h.operations.request(http.MethodPut, "/v1/operations/settings/scheduler", map[string]int{"maxConcurrentRuns": 1, "maxConcurrentToolRuns": 4}, http.StatusOK, map[string]string{"If-Match": etag})
	pids, owners, names := []int{}, []string{}, []string{}
	for index := range 4 / slots {
		name, owner := fmt.Sprintf("memory-runtime-%d", index), fmt.Sprintf("memory-%d-%d", os.Getpid(), index)
		process := h.startRuntimeSlots(name, "local/memory", true, slots,
			"--workspace-storage", "local", "--workspace-work-root", filepath.Join(h.temporary, name+"-workspace"),
			"--podman-enabled", "true", "--podman-image", image, "--podman-owner", owner)
		pids, owners, names = append(pids, process.cmd.Process.Pid), append(owners, owner), append(names, name)
	}
	runs := []string{}
	for range 4 {
		data := h.operations.request(http.MethodPost, "/v1/runs", map[string]any{"workflow": "placement-probe@1", "parameters": map[string]string{"target": target}, "artifacts": map[string]any{}, "labels": map[string]string{"local": "memory"}}, http.StatusAccepted, map[string]string{"Idempotency-Key": "memory-" + randomHex(t, 8)})
		var run runCreateResponse
		h.decode(t, data, &run)
		runs = append(runs, run.RunID)
	}
	count := func(pattern string) int {
		files, err := filepath.Glob(filepath.Join(workers, pattern))
		if err != nil {
			t.Fatal(err)
		}
		return len(files)
	}
	snapshot := func() []controlplane.RuntimeAgentObservation {
		var value struct {
			RuntimeAgents []controlplane.RuntimeAgentObservation `json:"runtimeAgents"`
		}
		h.decode(t, h.operations.request(http.MethodGet, "/v1/operations/snapshot", nil, http.StatusOK, nil), &value)
		return value.RuntimeAgents
	}
	wait(func() bool { return count("ready-*.json") == 4 })
	agents := snapshot()
	busy := 0
	for _, agent := range agents {
		busy += agent.BusySlots
		if agent.AllocationSlots != slots {
			t.Fatal("measurement capacity drift")
		}
	}
	if len(agents) != 4/slots || busy != 4 {
		t.Fatal("measurement did not start four concurrent allocations")
	}
	server, err := os.ReadFile(h.server.cmd.Path)
	if err != nil {
		t.Fatal(err)
	}
	record("ready", map[string]any{"runtime_pids": pids, "podman_owners": owners, "agents": agents, "units_per_worker": rounds, "bytes_per_unit": 65536, "server_binary_sha256": fmt.Sprintf("%x", sha256.Sum256(server))})
	wait(func() bool { _, err := os.Stat(filepath.Join(directory, "load")); return err == nil })
	before := snapshot()
	wait(func() bool { return count("result-*.json") == 4 })
	after := snapshot()
	for _, earlier := range before {
		progressed := false
		for _, later := range after {
			if later.InstanceID == earlier.InstanceID && later.LastAcceptedHeartbeat != nil && earlier.LastAcceptedHeartbeat != nil && later.LastAcceptedHeartbeat.After(*earlier.LastAcceptedHeartbeat) {
				progressed = true
			}
		}
		if !progressed {
			t.Fatal("a Runtime heartbeat made no progress during the CPU load")
		}
	}
	record("computed", map[string]any{"heartbeats_before": before, "heartbeats_after": after, "heartbeat_progress": true})
	wait(func() bool { _, err := os.Stat(filepath.Join(directory, "release")); return err == nil })
	for _, run := range runs {
		h.complete(t, run, "local/memory")
	}
	wait(func() bool {
		free := 0
		for _, agent := range snapshot() {
			free += agent.AvailableSlots
		}
		return free == 4
	})
	for _, name := range names {
		h.stopRuntime(name)
	}
	record("released", map[string]any{"completed_runs": 4, "free_slots_before_stop": 4, "runtimes_stopped": len(pids)})
}
