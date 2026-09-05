package projectworkflows

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	liveRunDeadline        = 30 * time.Minute
	liveReleaseDeadline    = 45 * time.Second
	liveValidatorDeadline  = 90 * time.Second
	liveMaxHTTPBody        = 4 << 20
	liveRuntimeRequestTime = 10 * time.Minute
)

type liveSettings struct {
	databaseURL  string
	gatewayURL   string
	gatewayToken string
	model        string
	workflows    []string
}

type liveArtifactRef struct {
	Namespace string  `json:"namespace"`
	Name      string  `json:"name"`
	Revision  *string `json:"revision,omitempty"`
}

type liveRunCreateResponse struct {
	RunID                string            `json:"runId"`
	State                string            `json:"state"`
	RuntimeLabels        []string          `json:"runtimeLabels"`
	Labels               map[string]string `json:"labels"`
	RuntimeConfiguration json.RawMessage   `json:"runtimeConfiguration"`
}

type liveRunStatus struct {
	RunID                  string                     `json:"runId"`
	Workflow               string                     `json:"workflow"`
	State                  string                     `json:"state"`
	RuntimeLabels          []string                   `json:"runtimeLabels"`
	Labels                 map[string]string          `json:"labels"`
	RuntimeConfiguration   json.RawMessage            `json:"runtimeConfiguration"`
	Cancellation           json.RawMessage            `json:"cancellation,omitempty"`
	Parameters             map[string]string          `json:"parameters,omitempty"`
	Inputs                 map[string]liveArtifactRef `json:"inputs,omitempty"`
	Attempts               []liveRunAttempt           `json:"attempts"`
	Transitions            []json.RawMessage          `json:"transitions"`
	Outputs                map[string]liveArtifactRef `json:"outputs"`
	EventCursor            json.RawMessage            `json:"eventCursor,omitempty"`
	ActiveStageExecutionID *string                    `json:"activeStageExecutionId,omitempty"`
	CreatedAt              json.RawMessage            `json:"createdAt,omitempty"`
	UpdatedAt              json.RawMessage            `json:"updatedAt,omitempty"`
	StartedAt              json.RawMessage            `json:"startedAt,omitempty"`
	FinishedAt             json.RawMessage            `json:"finishedAt,omitempty"`
}

type liveRunAttempt struct {
	StageExecutionID     string                        `json:"stageExecutionId"`
	Stage                string                        `json:"stage"`
	Objective            string                        `json:"objective,omitempty"`
	Attempt              int                           `json:"attempt"`
	PreviousExecutionID  *string                       `json:"previousExecutionId,omitempty"`
	ExecutionConfig      json.RawMessage               `json:"executionConfig"`
	State                string                        `json:"state"`
	Result               *contracts.StageContentResult `json:"result,omitempty"`
	Termination          *runstore.StageTermination    `json:"termination,omitempty"`
	Metrics              *telemetry.Summary            `json:"metrics,omitempty"`
	Diagnostics          json.RawMessage               `json:"diagnostics,omitempty"`
	Plan                 json.RawMessage               `json:"plan,omitempty"`
	RuntimeConfiguration json.RawMessage               `json:"runtimeConfiguration,omitempty"`
	CreatedAt            json.RawMessage               `json:"createdAt,omitempty"`
	UpdatedAt            json.RawMessage               `json:"updatedAt,omitempty"`
	PlannerStartedAt     json.RawMessage               `json:"plannerStartedAt,omitempty"`
	TerminalAt           json.RawMessage               `json:"terminalAt,omitempty"`
}

type liveAttemptEvidence struct {
	Stage              string              `json:"stage"`
	Attempt            int                 `json:"attempt"`
	State              string              `json:"state"`
	ResultOutcome      string              `json:"resultOutcome,omitempty"`
	ResultErrorCode    string              `json:"resultErrorCode,omitempty"`
	ResultRetryable    bool                `json:"resultRetryable,omitempty"`
	TerminationOutcome string              `json:"terminationOutcome,omitempty"`
	TerminationCode    string              `json:"terminationCode,omitempty"`
	TerminationPhase   string              `json:"terminationPhase,omitempty"`
	TerminationRetry   bool                `json:"terminationRetryable,omitempty"`
	ReportsComplete    bool                `json:"reportsComplete"`
	ModelCalls         int64               `json:"modelCalls"`
	ToolCalls          int64               `json:"toolCalls"`
	ToolFailures       int64               `json:"toolFailures"`
	TotalTokens        int64               `json:"totalTokens"`
	WorkerErrors       []liveErrorEvidence `json:"workerErrors,omitempty"`
}

type liveErrorEvidence struct {
	Code      string `json:"code"`
	Type      string `json:"type,omitempty"`
	Retryable *bool  `json:"retryable,omitempty"`
}

type liveWorkflowEvidence struct {
	Workflow string                `json:"workflow"`
	State    string                `json:"state"`
	Attempts []liveAttemptEvidence `json:"attempts"`
	Failures []Failure             `json:"failures"`
	Files    map[string][]byte     `json:"-"`
}

type liveEvaluationEvidence struct {
	SchemaVersion string                 `json:"schemaVersion"`
	ModelSHA256   string                 `json:"modelSha256"`
	StartedAt     time.Time              `json:"startedAt"`
	FinishedAt    time.Time              `json:"finishedAt"`
	Workflows     []liveWorkflowEvidence `json:"workflows"`
}

type liveStack struct {
	ctx            context.Context
	repositoryRoot string
	databaseURL    string
	publicBaseURL  string
	runtimeBaseURL string
	publicToken    string
	client         *http.Client
	controlClient  *http.Client
	server         *liveProcess
	runtime        *liveProcess
	workRoot       string
	workspaceRoot  string
	pool           *pgxpool.Pool
}

func TestLiveProjectWorkflows(t *testing.T) {
	settings, configured := loadLiveSettings(t)
	if !configured {
		t.Skip("live project Workflow environment is not configured")
	}
	for _, executable := range []string{"vacuum", "likec4"} {
		if _, err := exec.LookPath(executable); err != nil {
			t.Fatalf("required live validator %s is unavailable", executable)
		}
	}

	digest := sha256.Sum256([]byte(settings.model))
	evidence := liveEvaluationEvidence{
		SchemaVersion: "1.0",
		ModelSHA256:   hex.EncodeToString(digest[:]),
		StartedAt:     time.Now().UTC(),
		Workflows:     make([]liveWorkflowEvidence, 0, len(settings.workflows)),
	}
	repositoryRoot := liveRepositoryRoot(t)
	defer func() {
		evidence.FinishedAt = time.Now().UTC()
		if !t.Failed() {
			return
		}
		location, err := persistLiveEvidence(repositoryRoot, evidence)
		if err != nil {
			t.Logf("bounded live evaluation evidence could not be persisted (%s)", safeErrorType(err))
			return
		}
		t.Logf("bounded live evaluation evidence: %s", location)
	}()

	stack := startLiveStack(t, settings)
	source, err := SourceArchive()
	if err != nil {
		t.Fatal("build live source fixture")
	}
	sourceRef, err := uploadLiveArtifact(
		stack, "live-project-source-"+liveRandomHex(t, 4), "application/zip", source,
	)
	if err != nil {
		t.Fatalf("upload live source fixture (%s)", safeErrorType(err))
	}

	for _, workflow := range settings.workflows {
		workflowEvidence := evaluateLiveWorkflow(t, stack, workflow, sourceRef)
		evidence.Workflows = append(evidence.Workflows, workflowEvidence)
		if len(workflowEvidence.Failures) > 0 {
			codes := make([]string, 0, len(workflowEvidence.Failures))
			for _, failure := range workflowEvidence.Failures {
				codes = append(codes, failure.Code)
			}
			sort.Strings(codes)
			t.Errorf("%s failed semantic predicates: %s", workflow, strings.Join(codes, ", "))
		}
	}
}

func loadLiveSettings(t *testing.T) (liveSettings, bool) {
	t.Helper()
	settings := liveSettings{
		databaseURL:  strings.TrimSpace(os.Getenv("CONTRACTOR_TEST_DATABASE_URL")),
		gatewayURL:   strings.TrimSpace(os.Getenv("CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL")),
		gatewayToken: os.Getenv("CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_TOKEN"),
		model:        strings.TrimSpace(os.Getenv("CONTRACTOR_WORKFLOWS_LIVE_MODEL")),
		workflows:    []string{"openapi-from-workspace@5", "likec4-from-workspace@5"},
	}
	if settings.databaseURL == "" || settings.gatewayURL == "" || settings.model == "" {
		return liveSettings{}, false
	}
	if settings.gatewayToken == "" {
		settings.gatewayToken = "unused-live-gateway-token"
	}
	parsed, err := url.Parse(settings.gatewayURL)
	if err != nil || parsed.Host == "" || parsed.User != nil ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		t.Fatal("CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL must be an HTTP(S) API base without credentials")
	}
	if len(settings.model) > 256 || strings.ContainsAny(settings.model, "\r\n\x00") {
		t.Fatal("CONTRACTOR_WORKFLOWS_LIVE_MODEL is invalid")
	}
	if selected := strings.TrimSpace(os.Getenv("CONTRACTOR_WORKFLOWS_LIVE_ONLY")); selected != "" {
		switch selected {
		case "openapi-from-workspace@5", "likec4-from-workspace@5":
			settings.workflows = []string{selected}
		default:
			t.Fatal("CONTRACTOR_WORKFLOWS_LIVE_ONLY is not a supported live Workflow")
		}
	}
	return settings, true
}

func startLiveStack(t *testing.T, settings liveSettings) *liveStack {
	t.Helper()
	repositoryRoot := liveRepositoryRoot(t)
	temporaryRoot := t.TempDir()
	setupContext, cancelSetup := context.WithTimeout(context.Background(), 2*time.Minute)
	t.Cleanup(cancelSetup)
	databaseURL := liveIsolatedDatabase(t, setupContext, settings.databaseURL)
	configRoot := filepath.Join(temporaryRoot, "configs")
	copyLiveConfiguration(t, repositoryRoot, configRoot, settings.model, settings.gatewayURL)
	serverBinary := filepath.Join(temporaryRoot, "contractor-server")
	runLiveChecked(t, repositoryRoot, nil, "go", "build", "-o", serverBinary, "./cmd/contractor-server")
	runLiveChecked(t, repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL": databaseURL,
	}, serverBinary, "migrate")

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal("initialize live evaluation CA")
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:project-workflows-live",
	})
	if err != nil {
		t.Fatal("issue live Control Plane certificate")
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "project-workflows-live-agent", leaf)
	if err != nil {
		t.Fatal("issue live Runtime Agent certificate")
	}

	publicAddress := liveFreeAddress(t)
	privateAddress := liveFreeAddress(t)
	runtimeAddress := liveFreeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	publicToken := "live-public-" + liveRandomHex(t, 16)
	userID := "project-live-user-" + liveRandomHex(t, 8)
	localAuthFile := writeLiveLocalAuth(t, temporaryRoot, userID)
	server := startLiveProcess(t, "Go Server", repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL":            databaseURL,
		"CONTRACTOR_CONFIG_ROOT":             configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       settings.gatewayToken,
		"CONTRACTOR_PUBLIC_USER_ID":          userID,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}, serverBinary, "serve",
		"--runtime-request-timeout="+liveRuntimeRequestTime.String(),
		"--planner-timeout="+liveRunDeadline.String(),
	)
	publicClient := &http.Client{Timeout: 15 * time.Second}
	waitForLiveHTTP(t, setupContext, server, publicClient, publicBaseURL+"/readyz")

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing; run 'cd runtime && uv sync --locked'")
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	workspaceRoot := filepath.Join(temporaryRoot, "runtime-project-workspaces")
	runtimeProcess := startLiveProcess(
		t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--workspace-storage", "local",
		"--workspace-work-root", workspaceRoot,
		"--request-timeout-seconds", fmt.Sprintf("%.0f", liveRuntimeRequestTime.Seconds()),
		"--shutdown-grace-seconds", "10",
	)
	controlClient := newLiveMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForLiveHTTP(t, setupContext, runtimeProcess, controlClient, runtimeBaseURL+"/healthz")
	pool, err := pgxpool.New(setupContext, databaseURL)
	if err != nil {
		t.Fatal("open live assertion database")
	}
	t.Cleanup(pool.Close)
	return &liveStack{
		ctx: context.Background(), repositoryRoot: repositoryRoot, databaseURL: databaseURL,
		publicBaseURL: publicBaseURL, runtimeBaseURL: runtimeBaseURL, publicToken: publicToken,
		client: publicClient, controlClient: controlClient, server: server, runtime: runtimeProcess,
		workRoot: workRoot, workspaceRoot: workspaceRoot, pool: pool,
	}
}

func evaluateLiveWorkflow(
	t *testing.T,
	stack *liveStack,
	workflow string,
	source liveArtifactRef,
) liveWorkflowEvidence {
	t.Helper()
	evidence := liveWorkflowEvidence{
		Workflow: workflow, Files: make(map[string][]byte), Failures: []Failure{},
	}
	runID, err := createLiveRun(stack, workflow, source)
	if err != nil {
		evidence.State = "creation_failed"
		evidence.Failures = append(evidence.Failures, Failure{Code: "run.create", Message: "Workflow Run could not be created"})
		return evidence
	}
	status, waitFailure := waitForLiveRun(stack, runID, liveRunDeadline)
	evidence.State = status.State
	evidence.Attempts = liveAttemptEvidenceFrom(stack, status.Attempts)
	if waitFailure != nil {
		evidence.Failures = append(evidence.Failures, *waitFailure)
	}
	if status.State != string(runstore.RunSucceeded) {
		evidence.Failures = append(evidence.Failures, Failure{
			Code: "run.terminal_state", Message: "Workflow Run did not reach succeeded",
		})
	}

	dependency, dependencyErr := readLiveRunArtifact(stack, runID, "analysis", "dependencies")
	project, projectErr := readLiveRunArtifact(stack, runID, "analysis", "project")
	if dependencyErr != nil {
		evidence.Failures = append(evidence.Failures, Failure{Code: "discovery.dependency_artifact", Message: "dependency report artifact is unavailable"})
	}
	if projectErr != nil {
		evidence.Failures = append(evidence.Failures, Failure{Code: "discovery.project_artifact", Message: "project report artifact is unavailable"})
	}
	if dependencyErr == nil && projectErr == nil {
		evidence.Failures = append(evidence.Failures, ScoreDiscovery(dependency, project).Failures...)
		evidence.Files["dependency-report.md"] = boundedLiveArtifact(dependency)
		evidence.Files["project-report.md"] = boundedLiveArtifact(project)
	}

	switch workflow {
	case "openapi-from-workspace@5":
		var document, report []byte
		var documentErr, reportErr error
		if status.State == string(runstore.RunSucceeded) {
			document, documentErr = downloadLiveOutput(stack, runID, "openapi")
			report, reportErr = downloadLiveOutput(stack, runID, "openapi_validation_report")
		} else {
			document, documentErr = readLiveRunArtifact(stack, runID, "openapi", "openapi")
			report, reportErr = readLiveRunArtifact(stack, runID, "openapi", "validation-report")
		}
		if documentErr != nil {
			evidence.Failures = append(evidence.Failures, Failure{Code: "openapi.output", Message: "OpenAPI output or Run draft is unavailable"})
		} else {
			evidence.Failures = append(evidence.Failures, ScoreOpenAPI(document).Failures...)
			validationContext, cancel := context.WithTimeout(context.Background(), liveValidatorDeadline)
			if ValidateOpenAPI(validationContext, document) != nil {
				evidence.Failures = append(evidence.Failures, Failure{Code: "openapi.validator", Message: "independent Vacuum validation failed"})
			}
			cancel()
			evidence.Files["openapi.yaml"] = boundedLiveArtifact(document)
		}
		if reportErr == nil {
			evidence.Files["openapi-validation-report.md"] = boundedLiveArtifact(report)
		}
	case "likec4-from-workspace@5":
		var document, report []byte
		var documentErr, reportErr error
		if status.State == string(runstore.RunSucceeded) {
			document, documentErr = downloadLiveOutput(stack, runID, "likec4")
			report, reportErr = downloadLiveOutput(stack, runID, "likec4_validation_report")
		} else {
			document, documentErr = readLiveRunArtifact(stack, runID, "likec4", "architecture")
			report, reportErr = readLiveRunArtifact(stack, runID, "likec4", "validation-report")
		}
		if documentErr != nil {
			evidence.Failures = append(evidence.Failures, Failure{Code: "likec4.output", Message: "LikeC4 output or Run draft is unavailable"})
		} else {
			evidence.Failures = append(evidence.Failures, ScoreLikeC4(document).Failures...)
			validationContext, cancel := context.WithTimeout(context.Background(), liveValidatorDeadline)
			if ValidateLikeC4(validationContext, document) != nil {
				evidence.Failures = append(evidence.Failures, Failure{Code: "likec4.validator", Message: "independent LikeC4 validation failed"})
			}
			cancel()
			evidence.Files["architecture.c4"] = boundedLiveArtifact(document)
		}
		if reportErr == nil {
			evidence.Files["likec4-validation-report.md"] = boundedLiveArtifact(report)
		}
	}
	if err := waitForLiveRelease(stack, liveReleaseDeadline); err != nil {
		evidence.Failures = append(evidence.Failures, Failure{Code: "runtime.release", Message: "Runtime slot did not return to a clean idle state"})
	}
	sort.Slice(evidence.Failures, func(i, j int) bool {
		return evidence.Failures[i].Code < evidence.Failures[j].Code
	})
	return evidence
}

func uploadLiveArtifact(
	stack *liveStack,
	name, mediaType string,
	data []byte,
) (liveArtifactRef, error) {
	target := stack.publicBaseURL + "/v1/artifacts/projects/" + url.PathEscape(name)
	request, err := http.NewRequest(http.MethodPut, target, bytes.NewReader(data))
	if err != nil {
		return liveArtifactRef{}, err
	}
	request.Header.Set("Authorization", "Bearer "+stack.publicToken)
	request.Header.Set("Content-Type", mediaType)
	request.Header.Set("If-None-Match", "*")
	response, err := stack.client.Do(request)
	if err != nil {
		return liveArtifactRef{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		return liveArtifactRef{}, fmt.Errorf("artifact upload returned status %d", response.StatusCode)
	}
	var payload struct {
		Artifact  liveArtifactRef `json:"artifact"`
		MediaType string          `json:"mediaType"`
		Size      int64           `json:"size"`
	}
	if err := decodeLiveResponse(response.Body, &payload); err != nil ||
		payload.Artifact.Revision == nil || payload.MediaType != mediaType || payload.Size != int64(len(data)) {
		return liveArtifactRef{}, errors.New("artifact upload response is invalid")
	}
	return payload.Artifact, nil
}

func createLiveRun(stack *liveStack, workflow string, source liveArtifactRef) (string, error) {
	body, err := json.Marshal(map[string]any{
		"workflow": workflow,
		"parameters": map[string]string{
			"objective": "Document the implemented API, architecture, data flows, and trust boundaries",
		},
		"artifacts": map[string]liveArtifactRef{"source": source},
	})
	if err != nil {
		return "", err
	}
	request, err := http.NewRequest(http.MethodPost, stack.publicBaseURL+"/v1/runs", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	request.Header.Set("Authorization", "Bearer "+stack.publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "live-"+strings.ReplaceAll(workflow, "@", "-v")+"-"+time.Now().UTC().Format("20060102T150405.000000000"))
	response, err := stack.client.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		return "", fmt.Errorf("Run creation returned status %d", response.StatusCode)
	}
	var payload liveRunCreateResponse
	if err := decodeLiveResponse(response.Body, &payload); err != nil ||
		payload.RunID == "" || payload.State != string(runstore.RunRunning) {
		return "", errors.New("Run creation response is invalid")
	}
	return payload.RunID, nil
}

func waitForLiveRun(
	stack *liveStack,
	runID string,
	deadline time.Duration,
) (liveRunStatus, *Failure) {
	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	var last liveRunStatus
	for {
		status, err := getLiveRun(ctx, stack, runID)
		if err == nil {
			last = status
			switch status.State {
			case string(runstore.RunSucceeded), string(runstore.RunFailed), string(runstore.RunCancelled):
				return status, nil
			}
		}
		for _, process := range []*liveProcess{stack.server, stack.runtime} {
			if exited, _ := process.exited(); exited {
				return last, &Failure{Code: "run.process_exit", Message: "a production process exited while the Run was active"}
			}
		}
		select {
		case <-ctx.Done():
			requestLiveCancellation(stack, runID)
			cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), liveReleaseDeadline)
			defer cleanupCancel()
			for cleanupContext.Err() == nil {
				status, err := getLiveRun(cleanupContext, stack, runID)
				if err == nil {
					last = status
					if status.State == string(runstore.RunCancelled) || status.State == string(runstore.RunFailed) {
						break
					}
				}
				time.Sleep(500 * time.Millisecond)
			}
			return last, &Failure{Code: "run.deadline", Message: "Workflow Run exceeded the 30-minute live evaluation deadline"}
		case <-ticker.C:
		}
	}
}

func getLiveRun(ctx context.Context, stack *liveStack, runID string) (liveRunStatus, error) {
	request, _ := http.NewRequestWithContext(
		ctx, http.MethodGet, stack.publicBaseURL+"/v1/runs/"+url.PathEscape(runID), nil,
	)
	request.Header.Set("Authorization", "Bearer "+stack.publicToken)
	response, err := stack.client.Do(request)
	if err != nil {
		return liveRunStatus{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return liveRunStatus{}, fmt.Errorf("Run status returned %d", response.StatusCode)
	}
	var status liveRunStatus
	if err := decodeLiveResponse(response.Body, &status); err != nil {
		return liveRunStatus{}, err
	}
	return status, nil
}

func requestLiveCancellation(stack *liveStack, runID string) {
	body := strings.NewReader(`{"reason":"live evaluation deadline exceeded"}`)
	request, _ := http.NewRequest(
		http.MethodPost,
		stack.publicBaseURL+"/v1/runs/"+url.PathEscape(runID)+"/cancel",
		body,
	)
	request.Header.Set("Authorization", "Bearer "+stack.publicToken)
	request.Header.Set("Content-Type", "application/json")
	response, err := stack.client.Do(request)
	if err == nil {
		response.Body.Close()
	}
}

func readLiveRunArtifact(
	stack *liveStack,
	runID, namespace, name string,
) ([]byte, error) {
	service := artifacts.NewService(artifacts.NewPostgresRepository(stack.pool))
	store, err := service.Run(runID)
	if err != nil {
		return nil, err
	}
	result, err := store.Read(context.Background(), contracts.ArtifactRef{
		Namespace: namespace, Name: name,
	})
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), result.Payload.Data...), nil
}

func downloadLiveOutput(stack *liveStack, runID, output string) ([]byte, error) {
	request, _ := http.NewRequest(
		http.MethodGet,
		stack.publicBaseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/"+url.PathEscape(output),
		nil,
	)
	request.Header.Set("Authorization", "Bearer "+stack.publicToken)
	response, err := stack.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("output returned status %d", response.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, liveMaxHTTPBody+1))
	if err != nil || len(data) > liveMaxHTTPBody {
		return nil, errors.New("output exceeded the live evaluation bound")
	}
	return data, nil
}

func waitForLiveRelease(stack *liveStack, deadline time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()
	for ctx.Err() == nil {
		request, _ := http.NewRequestWithContext(
			ctx, http.MethodGet, stack.runtimeBaseURL+"/readyz", nil,
		)
		response, err := stack.controlClient.Do(request)
		if err == nil {
			var payload struct {
				State string `json:"state"`
			}
			if response.StatusCode == http.StatusOK {
				_ = json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&payload)
			}
			response.Body.Close()
			if payload.State == "idle" && liveWorkRootEmpty(stack.workRoot) &&
				liveWorkRootEmpty(stack.workspaceRoot) {
				return nil
			}
		}
		if exited, _ := stack.runtime.exited(); exited {
			return errors.New("Runtime process exited")
		}
		time.Sleep(250 * time.Millisecond)
	}
	return errors.New("Runtime release deadline exceeded")
}

func liveAttemptEvidenceFrom(stack *liveStack, attempts []liveRunAttempt) []liveAttemptEvidence {
	result := make([]liveAttemptEvidence, 0, len(attempts))
	repository := telemetry.NewRepository(stack.pool)
	for _, attempt := range attempts {
		current := liveAttemptEvidence{
			Stage: attempt.Stage, Attempt: attempt.Attempt, State: attempt.State,
		}
		if attempt.Result != nil {
			current.ResultOutcome = string(attempt.Result.Outcome)
			if attempt.Result.Error != nil {
				current.ResultErrorCode = attempt.Result.Error.Code
				current.ResultRetryable = attempt.Result.Error.Retryable
			}
		}
		if attempt.Termination != nil {
			current.TerminationOutcome = string(attempt.Termination.Outcome)
			current.TerminationCode = attempt.Termination.Code
			current.TerminationPhase = string(attempt.Termination.Phase)
			current.TerminationRetry = attempt.Termination.Retryable
		}
		if attempt.Metrics != nil {
			current.ReportsComplete = attempt.Metrics.ReportsComplete
			current.ModelCalls = attempt.Metrics.ModelCalls
			current.ToolCalls = attempt.Metrics.ToolCalls
			current.ToolFailures = attempt.Metrics.ToolFailures
			current.TotalTokens = attempt.Metrics.TotalTokens
		}
		if metrics, err := repository.GetStageMetrics(
			context.Background(), attempt.StageExecutionID,
		); err == nil {
			workerNames := make([]string, 0, len(metrics.Metrics.Workers))
			for name := range metrics.Metrics.Workers {
				workerNames = append(workerNames, name)
			}
			sort.Strings(workerNames)
			for _, name := range workerNames {
				for _, executionError := range metrics.Metrics.Workers[name].Errors {
					current.WorkerErrors = append(current.WorkerErrors, liveErrorEvidence{
						Code: executionError.Code, Type: safeLiveExecutionErrorType(executionError.Message),
						Retryable: executionError.Retryable,
					})
				}
			}
		}
		result = append(result, current)
	}
	return result
}

func safeLiveExecutionErrorType(message string) string {
	open := strings.LastIndexByte(message, '(')
	if open < 0 || !strings.HasSuffix(message, ")") {
		return ""
	}
	candidate := message[open+1 : len(message)-1]
	if candidate == "" || len(candidate) > 128 {
		return ""
	}
	for _, character := range candidate {
		if character < 'a' || character > 'z' {
			if character < 'A' || character > 'Z' {
				if character < '0' || character > '9' {
					if character != '_' && character != '.' {
						return ""
					}
				}
			}
		}
	}
	return candidate
}

func decodeLiveResponse(reader io.Reader, target any) error {
	decoder := json.NewDecoder(io.LimitReader(reader, liveMaxHTTPBody+1))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func boundedLiveArtifact(data []byte) []byte {
	if len(data) > liveMaxHTTPBody {
		return append([]byte(nil), data[:liveMaxHTTPBody]...)
	}
	return append([]byte(nil), data...)
}

func writeLiveLocalAuth(t *testing.T, root, userID string) string {
	t.Helper()
	hash, err := auth.HashPassword([]byte("contractor live evaluation password"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := auth.BootstrapYAML(userID, "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "local-auth.yaml")
	if err := os.WriteFile(path, document, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func persistLiveEvidence(repositoryRoot string, evidence liveEvaluationEvidence) (string, error) {
	root := filepath.Join(repositoryRoot, ".local", "eval-results")
	name := evidence.FinishedAt.UTC().Format("20060102T150405Z") + "-" + evidence.ModelSHA256[:12]
	directory := filepath.Join(root, name)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return "", err
	}
	summary, err := json.MarshalIndent(evidence, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(directory, "summary.json"), append(summary, '\n'), 0o600); err != nil {
		return "", err
	}
	for _, workflow := range evidence.Workflows {
		prefix := strings.TrimSuffix(strings.ReplaceAll(workflow.Workflow, "@", "-"), "-1")
		for name, data := range workflow.Files {
			if len(data) > liveMaxHTTPBody || strings.Contains(name, string(filepath.Separator)) {
				continue
			}
			if err := os.WriteFile(filepath.Join(directory, prefix+"-"+name), data, 0o600); err != nil {
				return "", err
			}
		}
	}
	relative, err := filepath.Rel(repositoryRoot, directory)
	if err != nil {
		return "", err
	}
	return relative, nil
}
