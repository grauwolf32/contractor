//go:build e2e

package e2e

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"go.yaml.in/yaml/v4"
)

// Inject a provider unload after the finding tool completed. Every repeated
// model request must contain exactly the same history, proving no tool replay.
type recoveryOutage struct {
	mu           sync.Mutex
	active       bool
	afterFinding bool
	calls        int
	payload      [sha256.Size]byte
	changed      bool
	observed     chan struct{}
	permanent    bool
}

func (o *recoveryOutage) intercept(w http.ResponseWriter, request map[string]any, encoded []byte) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.active || o.afterFinding && len(findingsToolResponses(request, "finding")) == 0 {
		return false
	}
	o.calls++
	if o.calls == 1 {
		o.payload = sha256.Sum256(encoded)
		close(o.observed)
	} else if o.payload != sha256.Sum256(encoded) {
		o.changed = true
	}
	w.Header().Set("Content-Type", "application/json")
	status := http.StatusBadRequest
	message := "Model is unloaded."
	if o.permanent {
		message = "Invalid messages: unsupported request format"
	} else if o.calls > 1 {
		status = http.StatusGatewayTimeout
		message = "upstream timeout"
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": message})
	return true
}
func (o *recoveryOutage) recover() { o.mu.Lock(); defer o.mu.Unlock(); o.active = false }

func TestGatewayRecoveryKeepsThreeQueuedRunsAcrossProcesses(t *testing.T) {
	outage := &recoveryOutage{active: true, afterFinding: true, observed: make(chan struct{})}
	h := startRecoveryHarness(t, outage, 3)
	project := createProjectResource(t, h.client, h.baseURL)
	body := recoveryFindingRunBody(t, h, project.ProjectID)
	var runs []string
	runs = append(runs, postProjectRun(t, h.client, h.baseURL, project.ProjectID, "recovery-first", body, false))
	select {
	case <-h.gateway.blockedRequest():
	case <-h.ctx.Done():
		t.Fatal("first Run did not reach model")
	}
	for i := range 2 {
		runs = append(runs, postProjectRun(t, h.client, h.baseURL, project.ProjectID, fmt.Sprint("recovery-queued-", i), body, false))
	}
	h.gateway.releaseBlockedRequest()
	select {
	case <-outage.observed:
	case <-h.ctx.Done():
		t.Fatal("outage not reached")
	}
	before := awaitRecoveryRunState(t, h, runs[0], "waiting")
	if before.Recovery == nil || before.Recovery.Code != "model_unavailable" || before.StartedAt == nil || len(before.Attempts) != 1 {
		t.Fatalf("waiting state=%+v", before)
	}
	for _, id := range runs[1:] {
		pending := awaitRecoveryRunState(t, h, id, "pending")
		if pending.StartedAt != nil {
			t.Fatal("queued Run acquired start timestamp")
		}
	}
	if got := findingsReceipts(t, h, runs[0]); len(got) != 1 {
		t.Fatalf("partial finding lost: %d", len(got))
	}
	// The next physical attempt is a 504, still inside the same invocation.
	for {
		waiting := awaitRecoveryRunState(t, h, runs[0], "waiting")
		if waiting.Recovery != nil && waiting.Recovery.Code == "gateway_timeout" {
			break
		}
		select {
		case <-h.ctx.Done():
			t.Fatal("504 was not classified as recoverable")
		case <-time.After(100 * time.Millisecond):
		}
	}
	// Advance only the recovery window in this isolated database; avoid waiting
	// five real minutes merely to exercise the manual-retry transition.
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if _, err := pool.Exec(h.ctx, `UPDATE gateway_recovery_routes SET automatic_until=clock_timestamp()-interval '1 second' WHERE blocked`); err != nil {
		t.Fatal(err)
	}
	paused := awaitRecoveryRunState(t, h, runs[0], "waiting")
	if paused.Recovery == nil || !paused.Recovery.RequiresRetry {
		t.Fatalf("missing manual retry: %+v", paused.Recovery)
	}
	outage.recover()
	request, _ := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/runs/"+runs[0]+"/retry-gateway", bytes.NewReader([]byte(`{}`)))
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	response, err := h.client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("manual recovery: HTTP %d", response.StatusCode)
	}
	for i, id := range runs {
		result := waitForFindingsRun(t, h, id, "succeeded")
		if len(result.Attempts) != 1 {
			t.Fatalf("Run %s replayed Stage: %d attempts", id, len(result.Attempts))
		}
		if i == 0 && result.Attempts[0].StageExecutionID != before.Attempts[0].StageExecutionID {
			t.Fatal("recovery changed invocation's stage identity")
		}
		if got := findingsReceipts(t, h, id); len(got) != 1 {
			t.Fatalf("finding replayed or lost: %d", len(got))
		}
	}
	outage.mu.Lock()
	defer outage.mu.Unlock()
	if outage.calls < 2 {
		t.Fatal("outage did not exercise both unload and timeout")
	}
	if outage.changed {
		t.Fatal("gateway retry replayed tools or changed model history")
	}
	if h.gateway.CompletedStages() != 3 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("script did not finish: %v", h.gateway.Failures())
	}
}

func TestGatewayRecoveryCancellationAndPermanentErrorAcrossProcesses(t *testing.T) {
	outage := &recoveryOutage{active: true, observed: make(chan struct{})}
	h := startRecoveryHarness(t, outage, 1)
	project := createProjectResource(t, h.client, h.baseURL)
	body := recoveryFindingRunBody(t, h, project.ProjectID)
	id := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "cancel-recovery", body, false)
	h.gateway.releaseBlockedRequest()
	awaitRecoveryRunState(t, h, id, "waiting")
	cancelBody := bytes.NewBufferString(`{"reason":"Cancel during model recovery"}`)
	request, _ := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/runs/"+id+"/cancel", cancelBody)
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	response, err := h.client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("cancel: HTTP %d", response.StatusCode)
	}
	waitForFindingsRun(t, h, id, "cancelled")
	outage.mu.Lock()
	outage.permanent = true
	outage.mu.Unlock()
	next := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "permanent-request-error", body, false)
	failed := waitForFindingsRun(t, h, next, "failed")
	if len(failed.Attempts) != 1 {
		t.Fatalf("permanent error replayed Stage: %d", len(failed.Attempts))
	}
	if failed.Recovery != nil {
		t.Fatal("permanent request error entered recovery")
	}
}

func awaitRecoveryRunState(t *testing.T, h *findingsHarness, id, want string) runStatus {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		var status runStatus
		if err := auditProgramTryGET(h.ctx, h.client, h.baseURL+"/v1/runs/"+id, &status); err == nil {
			if status.State == want {
				return status
			}
			if status.State == "failed" || status.State == "cancelled" || status.State == "succeeded" {
				t.Fatalf("Run ended before %s: %+v", want, status)
			}
		}
		select {
		case <-h.ctx.Done():
			t.Fatal("waiting for Run state", want)
		case <-ticker.C:
		}
	}
}

// Use the existing ordinary finding fixture so recovery coverage is independent
// of production finding interfaces and source-review workflows.
func startRecoveryHarness(t *testing.T, outage *recoveryOutage, count int) *findingsHarness {
	t.Helper()
	stages := make([]domainGatewayStage, count)
	for index := range stages {
		stage := ordinaryFindingsProducerStage("recovery")
		tools := make([]string, 0, len(stage.tools))
		for _, name := range stage.tools {
			if name != "list_skills" && name != "load_skill" && name != "load_skill_resource" {
				tools = append(tools, name)
			}
		}
		stage.tools = tools
		stages[index] = stage
	}
	return startConfiguredFindingsHarness(t, stages, func(root string) {
		// Skill preparation requires its own Runtime allocation. Leave it out
		// so all queued Runs can finish initialization during the model outage.
		path := filepath.Join(root, "agent-templates", "fixture-ordinary-findings.yaml")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var template map[string]any
		if err := yaml.Unmarshal(data, &template); err != nil {
			t.Fatal(err)
		}
		delete(template["spec"].(map[string]any), "skills")
		data, err = yaml.Marshal(template)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}, func(h *findingsHarness) { h.gateway.outage = outage })
}

func recoveryFindingRunBody(t *testing.T, h *findingsHarness, projectID string) []byte {
	t.Helper()
	sourcePayload := auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)})
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "sources", "recovery-source", "application/zip", sourcePayload)
	openAPI := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "openapi", "recovery-openapi", "application/yaml", []byte(findingsOpenAPI))
	return ordinaryFindingProducerBody(t, h, projectID, source, sourcePayload, openAPI, []byte(findingsOpenAPI))
}
