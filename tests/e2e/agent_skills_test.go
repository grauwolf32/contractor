//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

type agentSkillPackageVariant struct {
	payload []byte
	digest  string
}

type agentSkillRunEvidence struct {
	lastAllocationID  string
	runtimeInstanceID string
}

func TestAgentSkillsMVPProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 260*time.Second)
	defer cancel()

	fixture := loadAgentSkillMVPFixture(t, repositoryRoot)
	gateway := newBlockedDomainGateway(llmGatewayToken, agentSkillGatewayStages(fixture))
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	packageA, packageB := stageAgentSkillPackageVariants(t, configRoot, fixture)

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := filepath.Join(temporaryRoot, "contractor-server")
	runChecked(t, repositoryRoot, nil, "go", "build", "-o", serverBinary, "./cmd/contractor-server")
	runChecked(t, repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL": isolateURL,
	}, serverBinary, "migrate")

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatalf("initialize test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:agent-skills-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "agent-skills-e2e-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	publicAddress := freeAddress(t)
	privateAddress := freeAddress(t)
	runtimeAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "agent-skills-e2e-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	startServer := func(name string) *childProcess {
		return startProcess(t, name, repositoryRoot, map[string]string{
			"CONTRACTOR_DATABASE_URL":            isolateURL,
			"CONTRACTOR_CONFIG_ROOT":             configRoot,
			"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
			"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
			"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
			"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
			"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
			"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
			"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
			"CONTRACTOR_PUBLIC_USER_ID":          userID,
			"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
			"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
			"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
		}, serverBinary, "serve")
	}

	publicClient := &http.Client{Timeout: 8 * time.Second}
	firstServer := startServer("Go Server initial Skill seed")
	waitForHTTP(t, ctx, firstServer, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	seedA := assertAgentSkillCatalogAPI(
		t, publicClient, publicBaseURL, fixture.SkillName, packageA, 1,
	)
	firstServer.stop(t)
	publicClient.CloseIdleConnections()

	server := startServer("Go Server restarted Skill seed")
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)
	restartedA := assertAgentSkillCatalogAPI(
		t, publicClient, publicBaseURL, fixture.SkillName, packageA, 1,
	)
	if seedA.Ref.Revision == nil || restartedA.Ref.Revision == nil ||
		*seedA.Ref.Revision != *restartedA.Ref.Revision {
		t.Fatalf("Server restart advanced the bundled Skill revision")
	}
	assertPrivateTLSRejectsUnauthenticated(
		t, caPaths.Certificate, privateBaseURL+"/private/v1/agents/register",
	)

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	assertNoSkillSpecificSurface(t, ctx, pool, publicClient, publicBaseURL, repositoryRoot)

	validatorBin, validatorLog := installDomainValidators(t, temporaryRoot)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Python Runtime Agent with native Agent Skills", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{
			"PATH":             validatorBin + string(os.PathListSeparator) + os.Getenv("PATH"),
			"PYTHONUNBUFFERED": "1",
		},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--request-timeout-seconds", "12",
		"--shutdown-grace-seconds", "5",
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	blockerInput := uploadInput(t, publicClient, publicBaseURL)
	blockerRunID := createRun(t, publicClient, publicBaseURL, blockerInput)
	select {
	case <-gateway.blockedRequest():
	case <-ctx.Done():
		t.Fatalf("wait for blocking Worker model request: %v", ctx.Err())
	}

	source := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "agent-skills-source", "application/zip",
		projectSourceArchive(t),
	)
	seed := uploadProjectArtifact(
		t, publicClient, publicBaseURL, "agent-skills-likec4-seed", "text/plain",
		[]byte(likeC4Seed),
	)
	inputs := map[string]artifactRef{"source": source, "existing_likec4": seed}
	oldRunID := createAgentSkillWorkflowRun(
		t, publicClient, publicBaseURL, "agent-skills-old-a", inputs,
	)
	store := runstore.NewPostgresStore(pool)
	assertPendingAgentSkillSelection(
		t, ctx, store, oldRunID, *seedA.Ref.Revision, packageA,
	)

	updatedB := putAgentSkillPackage(
		t, publicClient, publicBaseURL, fixture.SkillName, packageB.payload, *seedA.Ref.Revision,
	)
	if updatedB.Revision == nil || *updatedB.Revision == *seedA.Ref.Revision {
		t.Fatal("ordinary Artifact PUT did not advance the Skill binding")
	}
	assertAgentSkillCatalogAPI(t, publicClient, publicBaseURL, fixture.SkillName, packageB, 2)
	oldExactURL := publicBaseURL + "/v1/artifacts/skills/" + url.PathEscape(fixture.SkillName) +
		"?revision=" + url.QueryEscape(*seedA.Ref.Revision)
	oldExact, oldExactMedia := download(t, publicClient, oldExactURL)
	if !bytes.Equal(oldExact, packageA.payload) || oldExactMedia != agentskills.MediaType {
		t.Fatal("pinned owner package A was not retained after package B update")
	}

	newRunID := createAgentSkillWorkflowRun(
		t, publicClient, publicBaseURL, "agent-skills-new-b", inputs,
	)
	assertPendingAgentSkillSelection(
		t, ctx, store, newRunID, *updatedB.Revision, packageB,
	)
	gateway.releaseBlockedRequest()

	blockerStatus := waitForRun(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, blockerRunID,
	)
	if len(blockerStatus.Attempts) != 1 || blockerStatus.Attempts[0].State != "succeeded" {
		t.Fatalf("blocking empty-Skill Run did not complete normally: %+v", blockerStatus.Attempts)
	}

	oldStatus := waitForDomainRun(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, oldRunID,
	)
	newStatus := waitForDomainRun(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, newRunID,
	)
	assertAgentSkillAttemptSequence(t, oldStatus, true)
	assertAgentSkillAttemptSequence(t, newStatus, false)
	oldEvidence := assertAgentSkillRunEvidence(
		t, ctx, store, publicClient, publicBaseURL, oldRunID,
		*seedA.Ref.Revision, packageA, fixture,
	)
	newEvidence := assertAgentSkillRunEvidence(
		t, ctx, store, publicClient, publicBaseURL, newRunID,
		*updatedB.Revision, packageB, fixture,
	)
	if oldEvidence.runtimeInstanceID != newEvidence.runtimeInstanceID {
		t.Fatalf("A/B Runs used different Runtime slots: %q != %q",
			oldEvidence.runtimeInstanceID, newEvidence.runtimeInstanceID)
	}

	reuseRunID := createWorkflowRun(
		t, publicClient, publicBaseURL, "artifact-copy@1", "agent-skills-empty-reuse", blockerInput,
	)
	reuseStatus := waitForRun(
		t, ctx, server, runtimeProcess, gateway, publicClient, publicBaseURL, reuseRunID,
	)
	reuseOutput := reuseStatus.Outputs["result"]
	if reuseOutput.Revision == nil {
		t.Fatalf("empty-Skill reuse Run has no exact output: %+v", reuseStatus.Outputs)
	}
	reuseBytes, reuseMedia := download(
		t, publicClient, publicBaseURL+"/v1/runs/"+url.PathEscape(reuseRunID)+"/outputs/result",
	)
	if string(reuseBytes) != e2eInput || reuseMedia != e2eMediaType {
		t.Fatalf("empty-Skill reuse output = (%q, %q)", reuseBytes, reuseMedia)
	}
	reuseEvidence := runAllocationEvidence(t, ctx, store, reuseRunID)
	if reuseEvidence.runtimeInstanceID != oldEvidence.runtimeInstanceID {
		t.Fatalf("empty-Skill Run did not reuse the Skill Runtime slot: %q != %q",
			reuseEvidence.runtimeInstanceID, oldEvidence.runtimeInstanceID)
	}
	waitForRuntimeReleased(
		t, ctx, runtimeProcess, controlClient, runtimeBaseURL,
		reuseEvidence.lastAllocationID, workRoot,
	)

	if gateway.CompletedStages() != 11 || gateway.Calls() != 82 || len(gateway.Failures()) != 0 {
		t.Fatalf("Agent Skill gateway stages/calls/failures = %d/%d/%v, want 11/82/none",
			gateway.CompletedStages(), gateway.Calls(), gateway.Failures())
	}
	assertAgentSkillGatewaySequence(t, gateway.Observations(), fixture)
	assertLikeC4ValidatorCount(t, validatorLog, 10)
	assertAgentSkillCanariesAbsent(
		t, fixture,
		firstServer.logs.redacted(publicToken, llmGatewayToken),
		server.logs.redacted(publicToken, llmGatewayToken),
		runtimeProcess.logs.redacted(publicToken, llmGatewayToken),
		string(mustJSON(t, oldStatus)), string(mustJSON(t, newStatus)),
		strings.Join(gateway.Failures(), "\n"),
	)
	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(firstServer.logs.redacted(), secret) ||
			strings.Contains(server.logs.redacted(), secret) ||
			strings.Contains(runtimeProcess.logs.redacted(), secret) {
			t.Fatal("process logs contain a configured secret")
		}
	}
}

func loadAgentSkillMVPFixture(t *testing.T, repositoryRoot string) agentSkillMVPFixture {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(
		repositoryRoot, "runtime", "tests", "fixtures", "agent_skills_mvp.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	var fixture agentSkillMVPFixture
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.SchemaVersion != "1.0" || fixture.SkillName != "likec4" ||
		fixture.ResourcePath != "references/cli.md" || fixture.PackageACanary == "" ||
		fixture.PackageBCanary == "" || fixture.PackageACanary == fixture.PackageBCanary ||
		!equalStringSlices(fixture.NativeToolSequence, []string{
			"list_skills", "load_skill", "load_skill_resource",
		}) {
		t.Fatalf("invalid Agent Skill MVP fixture: %+v", fixture)
	}
	return fixture
}

func stageAgentSkillPackageVariants(
	t *testing.T,
	configRoot string,
	fixture agentSkillMVPFixture,
) (agentSkillPackageVariant, agentSkillPackageVariant) {
	t.Helper()
	source := filepath.Join(configRoot, "skills", fixture.SkillName)
	resource := filepath.Join(source, filepath.FromSlash(fixture.ResourcePath))
	data, err := os.ReadFile(resource)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, []byte("\n<!-- "+fixture.PackageACanary+" -->\n")...)
	if err := os.WriteFile(resource, data, 0o600); err != nil {
		t.Fatal(err)
	}
	packageA := packageAgentSkillVariant(t, source)

	variantRoot := filepath.Join(t.TempDir(), fixture.SkillName)
	if err := os.CopyFS(variantRoot, os.DirFS(source)); err != nil {
		t.Fatal(err)
	}
	variantResource := filepath.Join(variantRoot, filepath.FromSlash(fixture.ResourcePath))
	variant, err := os.ReadFile(variantResource)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Count(variant, []byte(fixture.PackageACanary)) != 1 {
		t.Fatal("package A fixture marker is not unique")
	}
	variant = bytes.Replace(
		variant, []byte(fixture.PackageACanary), []byte(fixture.PackageBCanary), 1,
	)
	if err := os.WriteFile(variantResource, variant, 0o600); err != nil {
		t.Fatal(err)
	}
	packageB := packageAgentSkillVariant(t, variantRoot)
	if packageA.digest == packageB.digest || bytes.Equal(packageA.payload, packageB.payload) {
		t.Fatal("Agent Skill package fixtures are not distinguishable")
	}
	return packageA, packageB
}

func packageAgentSkillVariant(t *testing.T, source string) agentSkillPackageVariant {
	t.Helper()
	payload, validated, err := agentskills.PackageDirectory(source)
	if err != nil {
		t.Fatal(err)
	}
	if validated.Manifest.Name != "likec4" || validated.Digest != digestAgentSkillPayload(payload) {
		t.Fatalf("invalid deterministic Skill package metadata: %+v", validated)
	}
	return agentSkillPackageVariant{payload: payload, digest: validated.Digest}
}

func digestAgentSkillPayload(payload []byte) string {
	digest := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func assertAgentSkillCatalogAPI(
	t *testing.T,
	client *http.Client,
	baseURL, name string,
	want agentSkillPackageVariant,
	wantVersions int,
) artifacts.Metadata {
	t.Helper()
	var page struct {
		Items []artifacts.Metadata `json:"items"`
		Page  json.RawMessage      `json:"page"`
	}
	getAuthenticatedJSON(t, client, baseURL+"/v1/artifacts?namespace=skills", &page)
	if len(page.Items) != 1 || page.Items[0].Ref.Namespace != "skills" ||
		page.Items[0].Ref.Name != name || page.Items[0].Ref.Revision == nil ||
		page.Items[0].MediaType != agentskills.MediaType || !page.Items[0].Current {
		t.Fatalf("bundled Skill Artifact list = %+v", page.Items)
	}
	var metadata artifacts.Metadata
	getAuthenticatedJSON(
		t, client, baseURL+"/v1/artifacts/skills/"+url.PathEscape(name)+"/metadata", &metadata,
	)
	if !sameAgentSkillArtifactRef(metadata.Ref, page.Items[0].Ref) ||
		metadata.MediaType != agentskills.MediaType ||
		metadata.Size != int64(len(want.payload)) || !metadata.Current || metadata.Frozen {
		t.Fatalf("bundled Skill metadata = %+v", metadata)
	}
	payload, mediaType := download(
		t, client, baseURL+"/v1/artifacts/skills/"+url.PathEscape(name),
	)
	if !bytes.Equal(payload, want.payload) || mediaType != agentskills.MediaType ||
		digestAgentSkillPayload(payload) != want.digest {
		t.Fatal("bundled Skill bytes, media type, or digest differ from the canonical package")
	}
	var versions struct {
		Items []artifacts.Metadata `json:"items"`
		Page  json.RawMessage      `json:"page"`
	}
	getAuthenticatedJSON(
		t, client, baseURL+"/v1/artifacts/skills/"+url.PathEscape(name)+"/versions?limit=100",
		&versions,
	)
	if len(versions.Items) != wantVersions ||
		!sameAgentSkillArtifactRef(versions.Items[0].Ref, metadata.Ref) ||
		!versions.Items[0].Current {
		t.Fatalf("bundled Skill version history = %+v, want %d versions", versions.Items, wantVersions)
	}
	return metadata
}

func getAuthenticatedJSON(t *testing.T, client *http.Client, target string, result any) {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, target, nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	decodeResponse(t, response, result)
}

func putAgentSkillPackage(
	t *testing.T,
	client *http.Client,
	baseURL, name string,
	payload []byte,
	expectedRevision string,
) contracts.ArtifactRef {
	t.Helper()
	request, err := http.NewRequest(
		http.MethodPut, baseURL+"/v1/artifacts/skills/"+url.PathEscape(name),
		bytes.NewReader(payload),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", agentskills.MediaType)
	request.Header.Set("If-Match", strconv.Quote(expectedRevision))
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	var result struct {
		Artifact  contracts.ArtifactRef `json:"artifact"`
		MediaType string                `json:"mediaType"`
		Size      int64                 `json:"size"`
	}
	decodeResponse(t, response, &result)
	if result.Artifact.Namespace != "skills" || result.Artifact.Name != name ||
		result.Artifact.Revision == nil || result.MediaType != agentskills.MediaType ||
		result.Size != int64(len(payload)) {
		t.Fatalf("Skill Artifact update response = %+v", result)
	}
	return result.Artifact
}

func createAgentSkillWorkflowRun(
	t *testing.T,
	client *http.Client,
	baseURL, idempotencyKey string,
	inputs map[string]artifactRef,
) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"workflow": "likec4-from-source@3",
		"parameters": map[string]string{
			"objective": "Model the implemented API and architecture boundaries",
		},
		"artifacts": inputs,
	})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, baseURL+"/v1/runs", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", idempotencyKey)
	response := do(t, client, request, http.StatusAccepted)
	defer response.Body.Close()
	var result struct {
		RunID                string          `json:"runId"`
		State                string          `json:"state"`
		Labels               json.RawMessage `json:"labels"`
		RuntimeConfiguration json.RawMessage `json:"runtimeConfiguration"`
	}
	decodeResponse(t, response, &result)
	if result.RunID == "" || result.State != string(runstore.RunInitializing) {
		t.Fatalf("create Agent Skill Run = %+v", result)
	}
	return result.RunID
}

func assertPendingAgentSkillSelection(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	runID, sourceRevision string,
	pkg agentSkillPackageVariant,
) {
	t.Helper()
	run, err := store.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if run.State != runstore.RunInitializing ||
		run.StateReason.Code != runstore.SkillInitializationPendingReason ||
		len(run.SkillSnapshot) != 1 {
		t.Fatalf("pending Agent Skill Run = %+v", run)
	}
	skill := run.SkillSnapshot[0]
	if skill.Name != "likec4" || skill.Source == nil || skill.Source.Revision == nil ||
		*skill.Source.Revision != sourceRevision || skill.SourceDigest != pkg.digest ||
		skill.SourceSize != int64(len(pkg.payload)) || skill.Artifact != nil ||
		skill.PackageDigest != "" || skill.ExpandedBytes != 0 {
		t.Fatalf("pending exact Skill selection = %+v", skill)
	}
}

func assertAgentSkillAttemptSequence(t *testing.T, status runStatus, retried bool) {
	t.Helper()
	wantStages := []string{"dependency_discovery", "project_discovery", "likec4_build", "likec4_validate"}
	wantAttempts := []int{1, 1, 1, 1}
	wantStates := []string{"succeeded", "succeeded", "succeeded", "succeeded"}
	if retried {
		wantStages = []string{
			"dependency_discovery", "project_discovery", "likec4_build", "likec4_build", "likec4_validate",
		}
		wantAttempts = []int{1, 1, 1, 2, 1}
		wantStates = []string{"succeeded", "succeeded", "failed", "succeeded", "succeeded"}
	}
	if status.State != "succeeded" || len(status.Attempts) != len(wantStages) {
		t.Fatalf("Agent Skill Run attempts = %+v", status.Attempts)
	}
	for index, attempt := range status.Attempts {
		if attempt.Stage != wantStages[index] || attempt.Attempt != wantAttempts[index] ||
			attempt.State != wantStates[index] || attempt.Metrics == nil ||
			!attempt.Metrics.ReportsComplete {
			t.Fatalf("Agent Skill attempt %d = %+v", index, attempt)
		}
	}
}

func assertAgentSkillRunEvidence(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	client *http.Client,
	baseURL, runID, sourceRevision string,
	pkg agentSkillPackageVariant,
	fixture agentSkillMVPFixture,
) agentSkillRunEvidence {
	t.Helper()
	run, err := store.GetRun(ctx, runID)
	if err != nil || run.State != runstore.RunSucceeded || len(run.SkillSnapshot) != 1 {
		t.Fatalf("terminal Agent Skill Run = (%+v, %v)", run, err)
	}
	skill := run.SkillSnapshot[0]
	if skill.Source == nil || skill.Source.Revision == nil || *skill.Source.Revision != sourceRevision ||
		skill.Artifact == nil || skill.Artifact.Revision == nil ||
		skill.SourceDigest != pkg.digest || skill.PackageDigest != pkg.digest ||
		skill.SourceSize != int64(len(pkg.payload)) || skill.ExpandedBytes <= 0 {
		t.Fatalf("initialized Agent Skill provenance = %+v", skill)
	}
	runSkillURL := fmt.Sprintf(
		"%s/v1/runs/%s/artifacts/skills/%s?revision=%s", baseURL,
		url.PathEscape(runID), url.PathEscape(fixture.SkillName),
		url.QueryEscape(*skill.Artifact.Revision),
	)
	runPackage, mediaType := download(t, client, runSkillURL)
	if !bytes.Equal(runPackage, pkg.payload) || mediaType != agentskills.MediaType ||
		digestAgentSkillPayload(runPackage) != pkg.digest {
		t.Fatal("RunScope Skill fork bytes, media type, or digest are inconsistent")
	}
	var lineage struct {
		Items []artifacts.LineageEdge `json:"items"`
		Page  json.RawMessage         `json:"page"`
	}
	getAuthenticatedJSON(
		t, client,
		strings.TrimSuffix(runSkillURL, "?revision="+url.QueryEscape(*skill.Artifact.Revision))+
			"/lineage?revision="+url.QueryEscape(*skill.Artifact.Revision),
		&lineage,
	)
	if len(lineage.Items) != 1 || lineage.Items[0].Kind != artifacts.LineageInputFork ||
		lineage.Items[0].SourceScope != artifacts.ScopeUser ||
		lineage.Items[0].Source.Revision == nil ||
		*lineage.Items[0].Source.Revision != sourceRevision ||
		lineage.Items[0].TargetScope != artifacts.ScopeRun ||
		!sameAgentSkillArtifactRef(lineage.Items[0].Target, *skill.Artifact) {
		t.Fatalf("RunScope Skill lineage = %+v", lineage.Items)
	}

	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) < 4 {
		t.Fatalf("Agent Skill StageExecutions = (%+v, %v)", executions, err)
	}
	evidence := agentSkillRunEvidence{}
	for _, execution := range executions {
		allocations, allocationErr := store.ListStageAllocations(ctx, execution.StageExecutionID)
		if allocationErr != nil || len(allocations) != 1 {
			t.Fatalf("allocations for %s/%d = (%+v, %v)",
				execution.StageName, execution.Attempt, allocations, allocationErr)
		}
		allocation := allocations[0]
		if evidence.runtimeInstanceID == "" {
			evidence.runtimeInstanceID = allocation.RuntimeAgentInstanceID
		}
		if allocation.RuntimeAgentInstanceID != evidence.runtimeInstanceID {
			t.Fatalf("Run changed Runtime slot: %q != %q",
				allocation.RuntimeAgentInstanceID, evidence.runtimeInstanceID)
		}
		evidence.lastAllocationID = allocation.AllocationID
		reports, reportErr := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
		if reportErr != nil || len(reports) != 1 || !reports[0].Report.Worker.Complete ||
			!reports[0].Report.Runtime.Complete {
			t.Fatalf("reports for %s/%d = (%+v, %v)",
				execution.StageName, execution.Attempt, reports, reportErr)
		}
		serialized := string(mustJSON(t, reports[0]))
		assertAgentSkillCanariesAbsent(t, fixture, serialized)
		switch execution.StageName {
		case "likec4_build", "likec4_validate":
			assertContentFreeAgentSkillMetrics(t, reports[0].Report.Worker, fixture)
		default:
			for _, tool := range fixture.NativeToolSequence {
				if _, exists := reports[0].Report.Worker.Metrics.Tools[tool]; exists {
					t.Fatalf("unskilled Stage %s reports native Skill tool %s", execution.StageName, tool)
				}
			}
		}
	}
	if evidence.lastAllocationID == "" || evidence.runtimeInstanceID == "" {
		t.Fatal("Agent Skill Run has no allocation evidence")
	}
	return evidence
}

func sameAgentSkillArtifactRef(left, right contracts.ArtifactRef) bool {
	if left.Namespace != right.Namespace || left.Name != right.Name {
		return false
	}
	if left.Revision == nil || right.Revision == nil {
		return left.Revision == nil && right.Revision == nil
	}
	return *left.Revision == *right.Revision
}

func assertContentFreeAgentSkillMetrics(
	t *testing.T,
	report contracts.ExecutionReport,
	fixture agentSkillMVPFixture,
) {
	t.Helper()
	wantArguments := map[string]map[string]any{
		"list_skills": {},
		"load_skill":  {"skill_name": fixture.SkillName},
		"load_skill_resource": {
			"skill_name": fixture.SkillName, "file_path": fixture.ResourcePath,
		},
	}
	for _, name := range fixture.NativeToolSequence {
		metrics, ok := report.Metrics.Tools[name]
		if !ok || metrics.Calls == nil || *metrics.Calls != 1 ||
			metrics.Succeeded == nil || *metrics.Succeeded != 1 ||
			metrics.Failed != nil && *metrics.Failed != 0 {
			t.Fatalf("native Skill metrics for %s = %+v", name, metrics)
		}
		matches := 0
		for _, call := range report.ToolCalls {
			if call.Tool != name {
				continue
			}
			matches++
			if call.Outcome != contracts.ToolCallSucceeded || call.Error != nil ||
				call.ArgumentsTruncated || call.ResultSizeBytes == nil || *call.ResultSizeBytes <= 0 {
				t.Fatalf("native Skill call %s = %+v", name, call)
			}
			if len(wantArguments[name]) == 0 {
				if len(call.Arguments) != 0 {
					t.Fatalf("native Skill call %s arguments = %+v", name, call.Arguments)
				}
			} else if string(mustJSON(t, call.Arguments)) != string(mustJSON(t, wantArguments[name])) {
				t.Fatalf("native Skill call %s arguments = %+v", name, call.Arguments)
			}
		}
		if matches != 1 {
			t.Fatalf("native Skill call records for %s = %d", name, matches)
		}
	}
}

func runAllocationEvidence(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	runID string,
) agentSkillRunEvidence {
	t.Helper()
	executions, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(executions) != 1 {
		t.Fatalf("Run allocations require one StageExecution: (%+v, %v)", executions, err)
	}
	allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 1 {
		t.Fatalf("Run allocations = (%+v, %v)", allocations, err)
	}
	return agentSkillRunEvidence{
		lastAllocationID:  allocations[0].AllocationID,
		runtimeInstanceID: allocations[0].RuntimeAgentInstanceID,
	}
}

func assertAgentSkillGatewaySequence(
	t *testing.T,
	observations []domainGatewayObservation,
	fixture agentSkillMVPFixture,
) {
	t.Helper()
	counts := make(map[string]int)
	for _, observation := range observations {
		counts[observation.Tool]++
	}
	for _, tool := range fixture.NativeToolSequence {
		if counts[tool] != 5 {
			t.Fatalf("gateway observed %s %d times, want 5", tool, counts[tool])
		}
	}
}

func assertLikeC4ValidatorCount(t *testing.T, path string, want int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Fields(string(data))
	if len(lines) != want {
		t.Fatalf("LikeC4 validator invocations = %d, want %d", len(lines), want)
	}
	for _, line := range lines {
		if line != "likec4" {
			t.Fatalf("unexpected validator invocation marker %q", line)
		}
	}
}

func assertAgentSkillCanariesAbsent(
	t *testing.T,
	fixture agentSkillMVPFixture,
	values ...string,
) {
	t.Helper()
	for _, value := range values {
		if strings.Contains(value, fixture.PackageACanary) ||
			strings.Contains(value, fixture.PackageBCanary) {
			t.Fatal("Agent Skill content canary escaped a model response or test assertion")
		}
	}
}

func assertNoSkillSpecificSurface(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	client *http.Client,
	baseURL, repositoryRoot string,
) {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, baseURL+"/v1/skills", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusNotFound {
		t.Fatalf("Skill-specific public route returned HTTP %d", response.StatusCode)
	}

	openAPI, err := os.ReadFile(filepath.Join(repositoryRoot, "api", "openapi", "contractor-public-v1.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(openAPI), "\n  /v1/skills") {
		t.Fatal("public OpenAPI contains a Skill-specific route")
	}
	entries, err := os.ReadDir(filepath.Join(repositoryRoot, "internal", "persistence", "migrations"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".sql") {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(
			repositoryRoot, "internal", "persistence", "migrations", entry.Name(),
		))
		if readErr != nil {
			t.Fatal(readErr)
		}
		lower := strings.ToLower(string(data))
		if strings.Contains(lower, "systemscope") || strings.Contains(lower, "system_scope") {
			t.Fatalf("migration %s contains a SystemScope", entry.Name())
		}
	}
	rows, err := pool.Query(ctx, `
SELECT table_name
FROM information_schema.tables
WHERE table_schema = current_schema()
ORDER BY table_name`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var table string
		if err := rows.Scan(&table); err != nil {
			t.Fatal(err)
		}
		if strings.Contains(strings.ToLower(table), "skill") {
			t.Fatalf("database contains Skill-specific table %q", table)
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
