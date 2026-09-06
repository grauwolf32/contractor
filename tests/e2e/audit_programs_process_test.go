//go:build e2e

package e2e

import (
	"archive/zip"
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
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/jackc/pgx/v5/pgxpool"
)

type auditProgramAudit struct {
	AuditID           string `json:"auditId"`
	ProjectID         string `json:"projectId"`
	State             string `json:"state"`
	Revision          uint64 `json:"revision"`
	CurrentRoundID    string `json:"currentRoundId,omitempty"`
	SubmittedRunCount int    `json:"submittedRunCount"`
	OutstandingRuns   int    `json:"outstandingRunCount"`
	Baseline          *struct {
		Standards []struct {
			Reference struct {
				Scheme  string `json:"scheme"`
				Version string `json:"version"`
			} `json:"reference"`
			Source struct {
				Revision string `json:"revision"`
			} `json:"source"`
			License struct {
				ID string `json:"id"`
			} `json:"license"`
			Catalog  auditProgramExactPackage `json:"catalog"`
			Retained auditProgramExactPackage `json:"retained"`
		} `json:"standards"`
		Inventory struct {
			StandardSelection *struct {
				Scope    string   `json:"scope"`
				Levels   []string `json:"levels"`
				EntryIDs []string `json:"entryIds"`
			} `json:"standardSelection,omitempty"`
		} `json:"inventory"`
	} `json:"baseline,omitempty"`
}

type auditProgramExactPackage struct {
	Artifact artifactRef `json:"artifact"`
	Digest   string      `json:"digest"`
}

type auditProgramReview struct {
	RequestID   string `json:"requestId"`
	SubjectKind string `json:"subjectKind"`
	Kind        string `json:"kind"`
	State       string `json:"state"`
	Revision    uint64 `json:"revision"`
}

type auditProgramItem struct {
	ItemID           string `json:"itemId"`
	ItemKey          string `json:"itemKey"`
	State            string `json:"state"`
	FinalDisposition string `json:"finalDisposition,omitempty"`
	AcceptedResult   *struct {
		Ref artifactRef `json:"ref"`
	} `json:"acceptedResult,omitempty"`
	Attempts []struct {
		RunID                 string `json:"runId,omitempty"`
		RunDeleted            bool   `json:"runDeleted"`
		CollectionDisposition string `json:"collectionDisposition,omitempty"`
	} `json:"attempts"`
	Origin struct {
		Standard *struct {
			Scheme           string   `json:"scheme"`
			Version          string   `json:"version"`
			MappingKey       string   `json:"mappingKey"`
			EntryIDs         []string `json:"entryIds"`
			EvidenceContract struct {
				ID      string `json:"id"`
				Version string `json:"version"`
			} `json:"evidenceContract"`
		} `json:"standard,omitempty"`
	} `json:"origin"`
}

type auditProgramCoverage struct {
	ItemKey  string `json:"itemKey"`
	Coverage struct {
		Status    string   `json:"status"`
		Requested []string `json:"requested"`
		Completed []string `json:"completed"`
		Gaps      []string `json:"gaps"`
	} `json:"coverage"`
}

type auditProgramReport struct {
	Status  string          `json:"status"`
	Machine json.RawMessage `json:"machine"`
	Summary string          `json:"summary"`
}

type auditProgramFinding struct {
	FindingID       string  `json:"findingId"`
	State           string  `json:"state"`
	Revision        uint64  `json:"revision"`
	AnalystVerdict  *string `json:"analystVerdict,omitempty"`
	AnalystSeverity *string `json:"analystSeverity,omitempty"`
	FirstProposal   struct {
		ClientKey string `json:"clientKey"`
		Document  struct {
			StandardRefs []struct {
				Scheme        string `json:"scheme"`
				Version       string `json:"version"`
				RequirementID string `json:"requirement_id"`
			} `json:"standard_refs"`
		} `json:"document"`
		Origin struct {
			RunID      string `json:"runId"`
			RunDeleted bool   `json:"runDeleted"`
			Audit      *struct {
				AuditID string `json:"auditId"`
			} `json:"audit,omitempty"`
		} `json:"origin"`
	} `json:"firstProposal"`
}

func TestAuditProgramsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	// One Runtime intentionally executes all fourteen Worker allocations in
	// order. Keep the deadline bounded but leave room for slower CI hosts.
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()

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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:audit-programs-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "audit-programs-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newBlockedDomainGateway(llmGatewayToken, auditProgramGatewayStages())
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "audit-programs-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	serverEnvironment := map[string]string{
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
	}
	startServer := func() *childProcess {
		return startProcess(t, "Go Server", repositoryRoot, serverEnvironment, serverBinary, "serve")
	}
	server := startServer()

	client := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
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
		"--request-timeout-seconds", "12",
		"--shutdown-grace-seconds", "5",
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	assertAuditProfilesCompatible(t, client, publicBaseURL)
	project := createProjectResource(t, client, publicBaseURL)
	sourcePayload := auditProgramSourceArchive(t)
	source := uploadProjectScopeArtifact(
		t, client, publicBaseURL, project.ProjectID,
		"sources", "audit-fixture", "application/zip", sourcePayload,
	)
	checklist := uploadProjectScopeArtifact(
		t, client, publicBaseURL, project.ProjectID,
		"checklists", "source-checklist", "application/yaml", readAuditFixture(t, "checklist.yaml"),
	)
	openAPI := uploadProjectScopeArtifact(
		t, client, publicBaseURL, project.ProjectID,
		"openapi", "audit-fixture", "application/yaml", readAuditFixture(t, "openapi.yaml"),
	)

	checklistAudit := runAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, project.ProjectID,
		"source-checklist", map[string]artifactRef{"source": source, "checklist": checklist},
		2, 1, 0, "",
	)
	checklistCoverage := getAuditProgramCoverage(t, client, publicBaseURL, checklistAudit.AuditID)
	if len(checklistCoverage) != 2 || checklistCoverage[0].Coverage.Status != "satisfied" ||
		checklistCoverage[1].Coverage.Status != "inconclusive" {
		t.Fatalf("checklist coverage is not truthful: %+v", checklistCoverage)
	}
	assertAuditProgramReport(t, client, publicBaseURL, checklistAudit.AuditID, "completed-with-gaps")
	deleteCollectedAuditRuns(t, ctx, client, publicBaseURL, checklistAudit.AuditID)

	openAPIAudit := runAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, project.ProjectID,
		"openapi-operation-trace", map[string]artifactRef{"source": source, "openapi": openAPI},
		2, 2, 0, "",
	)
	openAPICoverage := getAuditProgramCoverage(t, client, publicBaseURL, openAPIAudit.AuditID)
	if len(openAPICoverage) != 2 {
		t.Fatalf("OpenAPI coverage count = %d, want 2", len(openAPICoverage))
	}
	allGaps := []string{}
	for _, row := range openAPICoverage {
		allGaps = append(allGaps, row.Coverage.Gaps...)
	}
	if !containsAuditGap(allGaps, "unsupported-callback") ||
		!containsAuditGap(openAPIAuditBaselineGaps(t, client, publicBaseURL, openAPIAudit.AuditID), "unsupported-webhook") {
		t.Fatalf("OpenAPI unsupported surfaces are missing: coverage=%v", allGaps)
	}
	assertAuditProgramReport(t, client, publicBaseURL, openAPIAudit.AuditID, "completed-with-gaps")
	deleteCollectedAuditRuns(t, ctx, client, publicBaseURL, openAPIAudit.AuditID)

	gateway.blockNextRequest()
	top10Audit := runAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, project.ProjectID,
		"owasp-top10-2025-source-risk", map[string]artifactRef{"source": source},
		10, 10, 2, "approve",
	)
	assertTop10AuditBaseline(t, client, publicBaseURL, top10Audit)
	top10Coverage := getAuditProgramCoverage(t, client, publicBaseURL, top10Audit.AuditID)
	assertTop10Coverage(t, top10Coverage)
	assertAuditProgramReport(t, client, publicBaseURL, top10Audit.AuditID, "completed-with-gaps")

	ordinaryASVS := runOrdinaryASVSFindingWorkflow(
		t, ctx, repositoryRoot, server, runtimeProcess, gateway, client, publicBaseURL,
		project.ProjectID, source, sourcePayload,
	)
	gateway.blockNextRequest()
	asvsAudit := runAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, project.ProjectID,
		"owasp-asvs-5-0-l1-source-review", map[string]artifactRef{"source": source},
		5, 4, 1, "not_applicable",
		func(audit auditProgramAudit, _ []auditProgramItem) {
			importOrdinaryFindingIntoAudit(
				t, client, publicBaseURL, audit.AuditID, ordinaryASVS.RunID, ordinaryASVS.Proposal,
			)
		},
	)
	assertASVSAuditBaseline(t, asvsAudit)
	asvsCoverage := getAuditProgramCoverage(t, client, publicBaseURL, asvsAudit.AuditID)
	assertASVSCoverage(t, asvsCoverage)
	assertAuditProgramReportSelection(t, client, publicBaseURL, asvsAudit.AuditID)
	backtrace := prepareASVSFindingBacktraceAfterRunDeletion(
		t, ctx, client, publicBaseURL, asvsAudit, ordinaryASVS.RunID,
	)
	replaceASVSCatalog(t, ctx, isolateURL, userID, configRoot)
	server.stop(t)
	server = startServer()
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
	assertASVSCatalogUnavailable(t, client, publicBaseURL)
	assertASVSFindingBacktrace(t, client, publicBaseURL, asvsAudit.AuditID, backtrace)

	if gateway.CompletedStages() != 18 || len(gateway.Failures()) != 0 {
		t.Fatalf("Audit gateway stages/failures = %d/%v, want 18/none", gateway.CompletedStages(), gateway.Failures())
	}
	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) || strings.Contains(runtimeProcess.logs.redacted(), secret) {
			t.Fatal("process logs contain a configured secret")
		}
	}
}

func auditProgramGatewayStages() []domainGatewayStage {
	tools := []string{
		"list_skills", "list_source_files", "load_skill", "load_skill_resource",
		"open_source_archive", "read_artifact", "read_source", "search_source",
		"read_audit_task", "submit_check_result",
	}
	result := func(name, assessment string, completed, gaps []string, evidence []map[string]string) domainGatewayStage {
		return domainGatewayStage{name: name, tools: tools, steps: []domainGatewayStep{
			toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
			toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
			toolGatewayStep("read_source", fixedArguments(map[string]any{
				"path": "app.py", "start_line": 1, "max_lines": 100,
			})),
			toolGatewayStep("submit_check_result", fixedArguments(map[string]any{
				"assessment": assessment,
				"summary":    "Deterministic fixture assessment based on source/app.py.",
				"completed":  completed,
				"gaps":       gaps,
				"evidence":   evidence,
			})),
			finalGatewayStep("Canonical Audit result package published", map[string]domainArtifactBinding{
				"result": {namespace: "audit-check", name: "result"},
			}),
		}}
	}
	stages := []domainGatewayStage{
		{name: "checklist/batch", tools: tools, steps: []domainGatewayStep{
			toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
			toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
			toolGatewayStep("read_source", fixedArguments(map[string]any{
				"path": "app.py", "start_line": 1, "max_lines": 100,
			})),
			toolGatewayStep("submit_check_result", fixedArguments(map[string]any{
				"results": []any{
					map[string]any{
						"assessment": "satisfied",
						"summary":    "Authorization call precedes the fixture object response.",
						"completed":  []string{"source-trace"},
						"gaps":       []string{},
						"evidence": []map[string]string{{
							"kind": "source-trace", "summary": "Authorization call precedes the fixture object response.",
						}},
					},
					map[string]any{
						"assessment": "inconclusive",
						"summary":    "The error path could not be established.",
						"completed":  []string{},
						"gaps":       []string{"missing-error-path"},
					},
				},
			})),
			finalGatewayStep("Canonical Audit batch result package published", map[string]domainArtifactBinding{
				"result": {namespace: "audit-check", name: "result"},
			}),
		}},
		result("openapi/deleteWidget", "satisfied", []string{"operation-resolution"}, []string{}, []map[string]string{{
			"kind": "source-trace", "summary": "DELETE operation maps to source/app.py.",
		}}),
		result("openapi/getWidget", "satisfied", []string{"operation-resolution"}, []string{}, []map[string]string{{
			"kind": "source-trace", "summary": "GET operation maps to source/app.py.",
		}}),
	}
	riskTools := append(append([]string{}, tools...), "finding", "write_artifact")
	top10Results := []struct {
		key        string
		assessment string
	}{
		{"A01:2025", "supported"}, {"A02:2025", "refuted"},
		{"A03:2025", "inconclusive"}, {"A04:2025", "not-tested"},
		{"A05:2025", "supported"}, {"A06:2025", "refuted"},
		{"A07:2025", "refuted"}, {"A08:2025", "inconclusive"},
		{"A09:2025", "supported"}, {"A10:2025", "not-tested"},
	}
	for _, candidate := range top10Results {
		completed, gaps, evidence := []string{}, []string{}, []map[string]string(nil)
		if candidate.assessment == "supported" || candidate.assessment == "refuted" {
			completed = []string{"observation"}
			evidence = []map[string]string{{
				"kind": "observation", "summary": "Bounded source observation from source/app.py.",
			}}
		} else {
			gaps = []string{"fixture-context-gap"}
		}
		stages = append(stages, domainGatewayStage{
			name: "top10/" + candidate.key, tools: riskTools, steps: []domainGatewayStep{
				toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
				toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
				toolGatewayStep("read_source", fixedArguments(map[string]any{
					"path": "app.py", "start_line": 1, "max_lines": 100,
				})),
				toolGatewayStep("submit_check_result", fixedArguments(map[string]any{
					"assessment": candidate.assessment,
					"summary":    "Bounded OWASP Top 10 fixture assessment based on source/app.py.",
					"completed":  completed,
					"gaps":       gaps,
					"evidence":   evidence,
				})),
				finalGatewayStep("Canonical Audit result package published", map[string]domainArtifactBinding{
					"result": {namespace: "audit-risk", name: "result"},
				}),
			},
		})
	}
	stages = append(stages, domainGatewayStage{
		name: "ordinary-run/asvs-reference-only", tools: riskTools, steps: []domainGatewayStep{
			toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
			toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
			toolGatewayStep("read_source", fixedArguments(map[string]any{
				"path": "app.py", "start_line": 1, "max_lines": 100,
			})),
			toolGatewayStep("finding", ordinaryASVSFindingArguments),
			toolGatewayStep("submit_check_result", fixedArguments(map[string]any{
				"assessment": "violated",
				"summary":    "Ordinary Run fixture result with a non-causal ASVS reference.",
				"completed":  []string{"observation"},
				"gaps":       []string{},
				"evidence": []map[string]string{{
					"kind": "observation", "summary": "Bounded ordinary-Run source observation.",
				}},
				"proposal_keys": []string{"ordinary-asvs-mapping"},
			})),
			finalGatewayStep("Ordinary Run ASVS-reference result published", map[string]domainArtifactBinding{
				"result": {namespace: "audit-asvs", name: "result"},
			}),
		},
	})
	asvsResults := []struct {
		key        string
		assessment string
		finding    bool
	}{
		{"v5.0.0-1.2.4", "violated", true},
		{"v5.0.0-1.2.5", "satisfied", false},
		{"v5.0.0-1.3.2", "inconclusive", false},
		{"v5.0.0-1.5.1", "not-tested", false},
	}
	for _, candidate := range asvsResults {
		completed, gaps, evidence := []string{}, []string{}, []map[string]string(nil)
		if candidate.assessment == "satisfied" || candidate.assessment == "violated" {
			completed = []string{"observation"}
			evidence = []map[string]string{{
				"kind": "observation", "summary": "Bounded ASVS source observation from source/app.py.",
			}}
		} else {
			gaps = []string{"fixture-context-gap"}
		}
		steps := []domainGatewayStep{
			toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
			toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
			toolGatewayStep("read_source", fixedArguments(map[string]any{
				"path": "app.py", "start_line": 1, "max_lines": 100,
			})),
		}
		proposalKeys := []string(nil)
		if candidate.finding {
			steps = append(steps, toolGatewayStep("finding", asvsFindingArguments))
			proposalKeys = []string{"asvs-database-injection"}
		}
		steps = append(steps,
			toolGatewayStep("submit_check_result", fixedArguments(map[string]any{
				"assessment":    candidate.assessment,
				"summary":       "Bounded ASVS fixture assessment based on source/app.py.",
				"completed":     completed,
				"gaps":          gaps,
				"evidence":      evidence,
				"proposal_keys": proposalKeys,
			})),
			finalGatewayStep("Canonical ASVS result package published", map[string]domainArtifactBinding{
				"result": {namespace: "audit-asvs", name: "result"},
			}),
		)
		stages = append(stages, domainGatewayStage{
			name: "asvs/" + candidate.key, tools: riskTools, steps: steps,
		})
	}
	return stages
}

func asvsFindingArguments(request map[string]any) (map[string]any, error) {
	source, exists := stageArtifact(request, "source")
	if !exists {
		return nil, fmt.Errorf("exact ASVS source artifact is absent")
	}
	return map[string]any{
		"client_key":    "asvs-database-injection",
		"title":         "Unparameterized database query",
		"description":   "The fixture constructs a database query from untrusted input without parameterization.",
		"subject":       map[string]string{"kind": "component", "key": "database-query"},
		"evidence_refs": []any{source},
		"standard_refs": []any{
			map[string]string{
				"scheme": "owasp-asvs", "version": "5.0.0",
				"requirement_id": "v5.0.0-1.2.4",
			},
			map[string]string{
				"scheme": "owasp-asvs", "version": "5.0.0",
				"requirement_id": "v5.0.0-1.2.5",
			},
		},
		"severity_suggestion": "high",
	}, nil
}

func ordinaryASVSFindingArguments(request map[string]any) (map[string]any, error) {
	source, exists := stageArtifact(request, "source")
	if !exists {
		return nil, fmt.Errorf("exact ordinary-Run ASVS source artifact is absent")
	}
	return map[string]any{
		"client_key":  "ordinary-asvs-mapping",
		"title":       "Ordinary Run finding with an ASVS mapping",
		"description": "This proposal exercises a non-causal standards mapping imported from an ordinary Run.",
		"subject":     map[string]string{"kind": "component", "key": "ordinary-database-query"},
		"evidence_refs": []any{
			source,
		},
		"standard_refs": []any{
			map[string]string{
				"scheme": "owasp-asvs", "version": "5.0.0",
				"requirement_id": "v5.0.0-1.2.5",
			},
		},
		"severity_suggestion": "medium",
	}, nil
}

func assertAuditProfilesCompatible(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	var page struct {
		Items []struct {
			Ref struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"ref"`
			ServerCompatible bool `json:"serverCompatible"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audit-profiles?limit=100", &page)
	wanted := map[string]bool{
		"source-checklist@1": false, "openapi-operation-trace@1": false,
		"owasp-top10-2025-source-risk@1":    false,
		"owasp-asvs-5-0-l1-source-review@1": false,
	}
	for _, profile := range page.Items {
		selector := profile.Ref.Name + "@" + profile.Ref.Version
		if _, exists := wanted[selector]; exists {
			wanted[selector] = profile.ServerCompatible
		}
	}
	for selector, compatible := range wanted {
		if !compatible {
			t.Fatalf("Audit profile %s is absent or incompatible: %+v", selector, page.Items)
		}
	}
}

type ordinaryASVSRun struct {
	RunID    string
	Proposal artifactRef
}

func runOrdinaryASVSFindingWorkflow(
	t *testing.T,
	ctx context.Context,
	repositoryRoot string,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, projectID string,
	source artifactRef,
	sourcePayload []byte,
) ordinaryASVSRun {
	t.Helper()
	sourceExact := exactContractRef(t, source)
	standardPayload, standard, err := auditstandards.PackageDirectory(
		filepath.Join(repositoryRoot, "configs", "audit-standards", "owasp-asvs-5.0.0"),
	)
	if err != nil {
		t.Fatalf("load exact ASVS fixture package: %v", err)
	}
	standardRef := uploadProjectScopeArtifact(
		t, client, baseURL, projectID, "fixtures", "ordinary-asvs-standard",
		auditstandards.MediaType, standardPayload,
	)
	inventory, err := auditdomain.BuildStandardMappingInventory(*standard, auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "standard",
		SourceRef: exactContractRef(t, standardRef),
		Scope: map[string]string{
			"objective": "Exercise a bounded ordinary-Run proposal with a non-causal ASVS reference.",
		},
		ApprovalRequirement: auditdomain.ApprovalNone,
		StandardSelection: &auditdomain.StandardSelection{
			Scope: "Ordinary Run ASVS mapping provenance fixture", Levels: []string{"1"},
			EntryIDs: []string{"v5.0.0-1.2.5"},
		},
	})
	if err != nil || len(inventory.Tasks) != 1 || len(inventory.ExecutionManifest.Items) != 1 ||
		inventory.SourceContentDigest == "" {
		t.Fatalf("build ordinary-Run ASVS fixture inventory: inventory=%+v err=%v", inventory, err)
	}
	task := inventory.Tasks[0]
	taskRef := uploadProjectScopeArtifact(
		t, client, baseURL, projectID, "fixtures", "ordinary-asvs-task",
		auditdomain.PackageMediaType, task.Package,
	)
	manifestItem := inventory.ExecutionManifest.Items[0]
	manifestItem.Ordinal = 0
	taskExact := exactContractRef(t, taskRef)
	manifestItem.TaskRef = &taskExact
	manifestItem.Inputs = []auditdomain.ExactInput{{
		Name: "source", Ref: sourceExact, Digest: auditProgramDigest(sourcePayload),
	}}
	manifest := auditdomain.ExecutionManifest{
		Schema: auditdomain.ExecutionManifestSchema,
		Items:  []auditdomain.ExecutionItem{manifestItem},
	}
	if err := auditdomain.ValidateDispatchExecutionManifest(manifest); err != nil {
		t.Fatalf("validate ordinary-Run execution manifest: %v", err)
	}
	manifestPayload, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatalf("encode ordinary-Run execution manifest: %v", err)
	}
	manifestRef := uploadProjectScopeArtifact(
		t, client, baseURL, projectID, "fixtures", "ordinary-asvs-execution-manifest",
		"application/json", manifestPayload,
	)
	body, err := json.Marshal(map[string]any{
		"workflow": "audit-asvs-source-verification@1",
		"artifacts": map[string]artifactRef{
			"task": taskRef, "execution_manifest": manifestRef, "source": source,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	runID := postProjectRun(
		t, client, baseURL, projectID, "ordinary-run-asvs-reference", body, false,
	)
	waitForDomainRun(t, ctx, server, runtimeProcess, gateway, client, baseURL, runID)

	var page struct {
		Items []struct {
			ClientKey string `json:"clientKey"`
			Proposal  struct {
				Ref artifactRef `json:"ref"`
			} `json:"proposal"`
			Origin struct {
				RunID string `json:"runId"`
				Audit *struct {
					AuditID string `json:"auditId"`
				} `json:"audit,omitempty"`
			} `json:"origin"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/runs/"+url.PathEscape(runID)+"/finding-proposals?limit=100", &page)
	if len(page.Items) != 1 || page.Items[0].ClientKey != "ordinary-asvs-mapping" ||
		page.Items[0].Origin.RunID != runID || page.Items[0].Origin.Audit != nil ||
		page.Items[0].Proposal.Ref.Revision == nil {
		t.Fatalf("ordinary Run proposal did not retain its non-Audit origin: %+v", page.Items)
	}
	return ordinaryASVSRun{RunID: runID, Proposal: page.Items[0].Proposal.Ref}
}

func exactContractRef(t *testing.T, ref artifactRef) contracts.ArtifactRef {
	t.Helper()
	if ref.Revision == nil || *ref.Revision == "" {
		t.Fatalf("artifact reference is not exact: %+v", ref)
	}
	revision := *ref.Revision
	return contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &revision}
}

func auditProgramDigest(payload []byte) string {
	digest := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func importOrdinaryFindingIntoAudit(
	t *testing.T,
	client *http.Client,
	baseURL, auditID, runID string,
	proposal artifactRef,
) {
	t.Helper()
	body, err := json.Marshal(map[string]any{"runId": runID, "proposal": proposal})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/finding-proposal-imports",
		bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	response := do(t, client, request, http.StatusCreated)
	response.Body.Close()
}

func runAuditProgram(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, projectID, profile string,
	inputs map[string]artifactRef,
	expectedItems, expectedRuns, expectedReviews int,
	reviewAction string,
	afterStart ...func(auditProgramAudit, []auditProgramItem),
) auditProgramAudit {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"profile": map[string]string{"name": profile, "version": "1"},
		"inputs":  inputs,
		"scope": map[string]string{
			"objective": "Exercise a bounded, non-certifying Audit fixture.",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/v1/projects/"+url.PathEscape(projectID)+"/audits",
		bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "audit-program-"+profile)
	response := do(t, client, request, http.StatusCreated)
	var draft auditProgramAudit
	decodeAuditProgramResponse(t, response, &draft)
	response.Body.Close()
	if draft.AuditID == "" || draft.State != "draft" || draft.Revision != 1 {
		t.Fatalf("create %s Audit = %+v", profile, draft)
	}

	startRequest, err := http.NewRequest(
		http.MethodPost, baseURL+"/v1/audits/"+url.PathEscape(draft.AuditID)+"/start", nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	startRequest.Header.Set("Authorization", "Bearer "+publicToken)
	startRequest.Header.Set("Idempotency-Key", "audit-program-start-"+profile)
	startRequest.Header.Set("If-Match", fmt.Sprintf("\"%d\"", draft.Revision))
	startResponse := do(t, client, startRequest, http.StatusOK)
	var started struct {
		Audit auditProgramAudit  `json:"audit"`
		Items []auditProgramItem `json:"items"`
	}
	decodeAuditProgramResponse(t, startResponse, &started)
	startResponse.Body.Close()
	if started.Audit.State != "active" || len(started.Items) != expectedItems {
		t.Fatalf("start %s Audit = %+v", profile, started)
	}
	if expectedReviews != 0 {
		decidePendingAuditItems(
			t, client, baseURL, draft.AuditID, expectedReviews, reviewAction,
		)
	}
	if len(afterStart) > 1 {
		t.Fatal("runAuditProgram accepts at most one after-start hook")
	}
	if len(afterStart) == 1 && afterStart[0] != nil {
		afterStart[0](started.Audit, started.Items)
	}
	gateway.releaseBlockedRequest()
	return waitForAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, baseURL, draft.AuditID, expectedRuns,
	)
}

func waitForAuditProgram(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, auditID string,
	expectedRuns int,
) auditProgramAudit {
	t.Helper()
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()
	for {
		var audit auditProgramAudit
		if auditProgramTryGET(ctx, client, baseURL+"/v1/audits/"+url.PathEscape(auditID), &audit) == nil {
			switch audit.State {
			case "completed":
				if audit.SubmittedRunCount != expectedRuns || audit.OutstandingRuns != 0 {
					var page struct {
						Items []auditProgramItem `json:"items"`
					}
					_ = auditProgramTryGET(
						ctx, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/items?limit=100", &page,
					)
					t.Fatalf("completed Audit counters = %+v; items=%+v", audit, page.Items)
				}
				return audit
			case "failed", "cancelled":
				t.Fatalf("Audit reached %s: %+v\nserver:\n%s\nruntime:\n%s\ngateway: %v",
					audit.State, audit, server.logs.redacted(publicToken, llmGatewayToken),
					runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
			}
		}
		for _, process := range []*childProcess{server, runtimeProcess} {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Audit was active: %v\n%s", process.name, processErr,
					process.logs.redacted(publicToken, llmGatewayToken))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Audit: %v\nserver:\n%s\nruntime:\n%s\ngateway: %v", ctx.Err(),
				server.logs.redacted(publicToken, llmGatewayToken),
				runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
		case <-ticker.C:
		}
	}
}

func decidePendingAuditItems(
	t *testing.T, client *http.Client, baseURL, auditID string,
	expected int, action string,
) {
	t.Helper()
	var page struct {
		Items []auditProgramReview `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/reviews?limit=100", &page)
	pending := 0
	for _, review := range page.Items {
		if review.State != "pending" {
			continue
		}
		if review.SubjectKind != "audit-item-action" || review.Kind != "requirement-applicability" ||
			review.Revision == 0 {
			t.Fatalf("unexpected pending Top 10 review: %+v", review)
		}
		body, err := json.Marshal(map[string]string{
			"action": action, "rationale": "Apply this exact owner decision to the selected requirement.",
		})
		if err != nil {
			t.Fatal(err)
		}
		request, err := http.NewRequest(
			http.MethodPost,
			baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/reviews/"+
				url.PathEscape(review.RequestID)+"/decisions",
			bytes.NewReader(body),
		)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Idempotency-Key", "audit-program-review-"+review.RequestID)
		request.Header.Set("If-Match", fmt.Sprintf("\"%d\"", review.Revision))
		response := do(t, client, request, http.StatusOK)
		response.Body.Close()
		pending++
	}
	if pending != expected {
		t.Fatalf("pending applicability reviews = %d, want %d; all=%+v", pending, expected, page.Items)
	}
}

func assertTop10AuditBaseline(
	t *testing.T, client *http.Client, baseURL string, audit auditProgramAudit,
) {
	t.Helper()
	if audit.Baseline == nil || len(audit.Baseline.Standards) != 1 {
		t.Fatalf("Top 10 Audit exact standard baseline = %+v", audit.Baseline)
	}
	standard := audit.Baseline.Standards[0]
	if standard.Reference.Scheme != "owasp-web-top10" || standard.Reference.Version != "2025" ||
		standard.Source.Revision != "66ebc4798d2ca72973967a20264bdeb70dcf0a13" ||
		standard.License.ID != "CC-BY-SA-4.0" || standard.Catalog.Digest == "" ||
		standard.Catalog.Digest != standard.Retained.Digest || standard.Catalog.Artifact.Revision == nil ||
		standard.Retained.Artifact.Revision == nil {
		t.Fatalf("Top 10 Audit did not retain exact licensed provenance: %+v", standard)
	}
	var detail struct {
		Standard struct {
			Digest       string `json:"digest"`
			EntryCount   int    `json:"entryCount"`
			MappingCount int    `json:"mappingCount"`
			Entries      []struct {
				ID string `json:"id"`
			} `json:"entries"`
		} `json:"standard"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audit-standards/owasp-web-top10/versions/2025", &detail)
	if detail.Standard.Digest != standard.Catalog.Digest || detail.Standard.EntryCount != 10 ||
		detail.Standard.MappingCount != 10 || len(detail.Standard.Entries) != 10 {
		t.Fatalf("Top 10 exact catalog projection = %+v", detail.Standard)
	}
}

func assertTop10Coverage(t *testing.T, rows []auditProgramCoverage) {
	t.Helper()
	if len(rows) != 10 {
		t.Fatalf("Top 10 coverage count = %d, want 10", len(rows))
	}
	want := map[string]string{
		"A01:2025": "violated", "A02:2025": "satisfied", "A03:2025": "inconclusive",
		"A04:2025": "not-tested", "A05:2025": "violated", "A06:2025": "satisfied",
		"A07:2025": "satisfied", "A08:2025": "inconclusive", "A09:2025": "violated",
		"A10:2025": "not-tested",
	}
	for _, row := range rows {
		status, exists := want[row.ItemKey]
		if !exists || row.Coverage.Status != status ||
			len(row.Coverage.Requested) != 1 || row.Coverage.Requested[0] != "observation" {
			t.Fatalf("Top 10 coverage row is not an exact mixed projection: %+v", row)
		}
		delete(want, row.ItemKey)
	}
	if len(want) != 0 {
		t.Fatalf("Top 10 coverage omitted categories: %+v", want)
	}
}

var asvsSelectedRequirementIDs = []string{
	"v5.0.0-1.2.4",
	"v5.0.0-1.2.5",
	"v5.0.0-1.3.2",
	"v5.0.0-1.5.1",
	"v5.0.0-2.1.1",
}

func assertASVSAuditBaseline(t *testing.T, audit auditProgramAudit) {
	t.Helper()
	if audit.Baseline == nil || len(audit.Baseline.Standards) != 1 ||
		audit.Baseline.Inventory.StandardSelection == nil {
		t.Fatalf("ASVS Audit exact baseline = %+v", audit.Baseline)
	}
	standard := audit.Baseline.Standards[0]
	selection := audit.Baseline.Inventory.StandardSelection
	if standard.Reference.Scheme != "owasp-asvs" || standard.Reference.Version != "5.0.0" ||
		standard.Source.Revision != "5cf9b032440be53ce345ab3c130fda46ba1ce7a2" ||
		standard.License.ID != "CC-BY-SA-4.0" || standard.Catalog.Digest == "" ||
		standard.Catalog.Digest != standard.Retained.Digest || standard.Catalog.Artifact.Revision == nil ||
		standard.Retained.Artifact.Revision == nil ||
		selection.Scope != "ASVS 5.0 Level 1 source and documentation pilot (5 requirements)" ||
		!equalStrings(selection.Levels, []string{"1"}) ||
		!equalStrings(selection.EntryIDs, asvsSelectedRequirementIDs) {
		t.Fatalf("ASVS Audit did not retain its exact selected authority: baseline=%+v", audit.Baseline)
	}
}

func assertASVSCoverage(t *testing.T, rows []auditProgramCoverage) {
	t.Helper()
	want := map[string]string{
		"v5.0.0-1.2.4": "violated",
		"v5.0.0-1.2.5": "satisfied",
		"v5.0.0-1.3.2": "inconclusive",
		"v5.0.0-1.5.1": "not-tested",
		"v5.0.0-2.1.1": "not-applicable",
	}
	if len(rows) != len(want) {
		t.Fatalf("ASVS coverage count = %d, want %d", len(rows), len(want))
	}
	for _, row := range rows {
		status, exists := want[row.ItemKey]
		if !exists || row.Coverage.Status != status || len(row.Coverage.Requested) != 1 ||
			row.Coverage.Requested[0] != "observation" {
			t.Fatalf("ASVS coverage row is not an exact selected projection: %+v", row)
		}
		if status == "not-applicable" &&
			(len(row.Coverage.Completed) != 0 || len(row.Coverage.Gaps) != 0) {
			t.Fatalf("human not-applicable decision inherited Worker evidence: %+v", row)
		}
		delete(want, row.ItemKey)
	}
	if len(want) != 0 {
		t.Fatalf("ASVS coverage omitted selected requirements: %+v", want)
	}
}

func assertAuditProgramReportSelection(
	t *testing.T, client *http.Client, baseURL, auditID string,
) {
	t.Helper()
	var report auditProgramReport
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/report", &report)
	var machine struct {
		Profile struct {
			StandardSelection *struct {
				Scope    string   `json:"scope"`
				Levels   []string `json:"levels"`
				EntryIDs []string `json:"entryIds"`
			} `json:"standardSelection"`
		} `json:"profile"`
		Baseline struct {
			StandardSelection *struct {
				Scope    string   `json:"scope"`
				Levels   []string `json:"levels"`
				EntryIDs []string `json:"entryIds"`
			} `json:"standardSelection"`
		} `json:"baseline"`
		Coverage struct {
			SelectedItems         int `json:"selectedItems"`
			ApplicableDenominator int `json:"applicableDenominator"`
			AssessedApplicable    int `json:"assessedApplicable"`
			Counts                struct {
				Satisfied     int `json:"satisfied"`
				Violated      int `json:"violated"`
				Inconclusive  int `json:"inconclusive"`
				NotTested     int `json:"notTested"`
				NotApplicable int `json:"notApplicable"`
			} `json:"counts"`
		} `json:"coverage"`
	}
	if err := json.Unmarshal(report.Machine, &machine); err != nil {
		t.Fatalf("decode ASVS machine report: %v", err)
	}
	profileSelection, baselineSelection := machine.Profile.StandardSelection, machine.Baseline.StandardSelection
	if report.Status != "ready" || profileSelection == nil || baselineSelection == nil ||
		profileSelection.Scope != baselineSelection.Scope ||
		!equalStrings(profileSelection.Levels, []string{"1"}) ||
		!equalStrings(profileSelection.EntryIDs, asvsSelectedRequirementIDs) ||
		!equalStrings(baselineSelection.EntryIDs, asvsSelectedRequirementIDs) ||
		machine.Coverage.SelectedItems != 5 || machine.Coverage.ApplicableDenominator != 4 ||
		machine.Coverage.AssessedApplicable != 2 || machine.Coverage.Counts.Satisfied != 1 ||
		machine.Coverage.Counts.Violated != 1 || machine.Coverage.Counts.Inconclusive != 1 ||
		machine.Coverage.Counts.NotTested != 1 || machine.Coverage.Counts.NotApplicable != 1 ||
		!strings.Contains(report.Summary, "Exact selected requirements: 5") ||
		!strings.Contains(report.Summary, "not a security or compliance certification") {
		t.Fatalf("ASVS report lost its honest selected denominator: report=%+v machine=%+v", report, machine)
	}
}

type asvsBacktraceFixture struct {
	CausalFindingID   string
	CausalRunID       string
	OrdinaryFindingID string
	OrdinaryRunID     string
}

func prepareASVSFindingBacktraceAfterRunDeletion(
	t *testing.T,
	ctx context.Context,
	client *http.Client,
	baseURL string,
	audit auditProgramAudit,
	ordinaryRunID string,
) asvsBacktraceFixture {
	t.Helper()
	var itemPage struct {
		Items []auditProgramItem `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(audit.AuditID)+"/items?limit=100", &itemPage)
	if len(itemPage.Items) != len(asvsSelectedRequirementIDs) {
		t.Fatalf("ASVS item count = %d, want %d", len(itemPage.Items), len(asvsSelectedRequirementIDs))
	}
	var causalItem *auditProgramItem
	for index := range itemPage.Items {
		item := &itemPage.Items[index]
		if item.Origin.Standard == nil || item.Origin.Standard.Scheme != "owasp-asvs" ||
			item.Origin.Standard.Version != "5.0.0" || item.Origin.Standard.MappingKey != item.ItemKey ||
			!equalStrings(item.Origin.Standard.EntryIDs, []string{item.ItemKey}) {
			t.Fatalf("ASVS item lost its causal standard origin: %+v", item)
		}
		if item.ItemKey == "v5.0.0-1.2.4" {
			causalItem = item
		}
		if item.ItemKey == "v5.0.0-2.1.1" &&
			(item.FinalDisposition != "not-applicable" || len(item.Attempts) != 0) {
			t.Fatalf("manual ASVS item unexpectedly executed a Worker: %+v", item)
		}
	}
	if causalItem == nil || causalItem.Origin.Standard.EvidenceContract.ID != "bounded-source-verification" ||
		causalItem.Origin.Standard.EvidenceContract.Version != "1" ||
		causalItem.FinalDisposition != "accepted-result" || len(causalItem.Attempts) != 1 ||
		causalItem.Attempts[0].RunID == "" {
		t.Fatalf("ASVS causal item is incomplete: %+v", causalItem)
	}

	var findingPage struct {
		Items []auditProgramFinding `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(audit.AuditID)+"/findings?limit=100", &findingPage)
	if len(findingPage.Items) != 2 {
		t.Fatalf("ASVS standard mappings multiplied findings: %+v", findingPage.Items)
	}
	var causalFinding, ordinaryFinding *auditProgramFinding
	for index := range findingPage.Items {
		finding := &findingPage.Items[index]
		switch finding.FirstProposal.ClientKey {
		case "asvs-database-injection":
			causalFinding = finding
		case "ordinary-asvs-mapping":
			ordinaryFinding = finding
		default:
			t.Fatalf("unexpected ASVS finding identity: %+v", finding)
		}
	}
	if causalFinding == nil || causalFinding.FirstProposal.Origin.Audit == nil ||
		causalFinding.FirstProposal.Origin.Audit.AuditID != audit.AuditID ||
		!hasASVSReference(*causalFinding, "v5.0.0-1.2.4") ||
		!hasASVSReference(*causalFinding, "v5.0.0-1.2.5") {
		t.Fatalf("ASVS finding did not retain causal and incidental references: %+v", causalFinding)
	}
	if ordinaryFinding == nil || ordinaryFinding.FirstProposal.Origin.Audit != nil ||
		ordinaryFinding.FirstProposal.Origin.RunID != ordinaryRunID ||
		len(ordinaryFinding.FirstProposal.Document.StandardRefs) != 1 ||
		!hasASVSReference(*ordinaryFinding, "v5.0.0-1.2.5") {
		t.Fatalf("ordinary finding was assigned a false ASVS Audit origin: %+v", ordinaryFinding)
	}

	reviewBody := bytes.NewReader([]byte(`{}`))
	reviewRequest, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/v1/audits/"+url.PathEscape(audit.AuditID)+"/findings/"+
			url.PathEscape(causalFinding.FindingID)+"/reviews",
		reviewBody,
	)
	if err != nil {
		t.Fatal(err)
	}
	reviewRequest.Header.Set("Authorization", "Bearer "+publicToken)
	reviewRequest.Header.Set("Content-Type", "application/json")
	reviewRequest.Header.Set("Idempotency-Key", "asvs-finding-review")
	reviewRequest.Header.Set("If-Match", fmt.Sprintf("\"%d\"", causalFinding.Revision))
	reviewResponse := do(t, client, reviewRequest, http.StatusCreated)
	var review auditProgramReview
	decodeAuditProgramResponse(t, reviewResponse, &review)
	reviewResponse.Body.Close()

	decisionBody, err := json.Marshal(map[string]string{
		"verdict": "true_positive", "severity": "high",
		"rationale": "The exact accepted ASVS check attempt supports this rating.",
	})
	if err != nil {
		t.Fatal(err)
	}
	decisionRequest, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/v1/audits/"+url.PathEscape(audit.AuditID)+"/reviews/"+
			url.PathEscape(review.RequestID)+"/decisions",
		bytes.NewReader(decisionBody),
	)
	if err != nil {
		t.Fatal(err)
	}
	decisionRequest.Header.Set("Authorization", "Bearer "+publicToken)
	decisionRequest.Header.Set("Content-Type", "application/json")
	decisionRequest.Header.Set("Idempotency-Key", "asvs-finding-decision")
	decisionRequest.Header.Set("If-Match", fmt.Sprintf("\"%d\"", review.Revision))
	decisionResponse := do(t, client, decisionRequest, http.StatusOK)
	decisionResponse.Body.Close()

	causalRunID := causalItem.Attempts[0].RunID
	deleteASVSBacktraceRun(t, ctx, client, baseURL, causalRunID)
	deleteASVSBacktraceRun(t, ctx, client, baseURL, ordinaryRunID)
	return asvsBacktraceFixture{
		CausalFindingID: causalFinding.FindingID, CausalRunID: causalRunID,
		OrdinaryFindingID: ordinaryFinding.FindingID, OrdinaryRunID: ordinaryRunID,
	}
}

func deleteASVSBacktraceRun(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, runID string,
) {
	t.Helper()
	waitForAuditRunDeletable(t, ctx, client, baseURL, runID)
	deleteRequest, err := http.NewRequest(http.MethodDelete, baseURL+"/v1/runs/"+url.PathEscape(runID), nil)
	if err != nil {
		t.Fatal(err)
	}
	deleteRequest.Header.Set("Authorization", "Bearer "+publicToken)
	deleteResponse := do(t, client, deleteRequest, http.StatusNoContent)
	deleteResponse.Body.Close()
}

func replaceASVSCatalog(
	t *testing.T, ctx context.Context, databaseURL, ownerID, configRoot string,
) {
	t.Helper()
	profilePath := filepath.Join(configRoot, "audit-profiles", "owasp-asvs-5.0-l1-source-review.yaml")
	packagePath := filepath.Join(configRoot, "audit-standards", "owasp-asvs-5.0.0")
	if err := os.Remove(profilePath); err != nil {
		t.Fatalf("remove staged ASVS profile: %v", err)
	}
	if err := os.RemoveAll(packagePath); err != nil {
		t.Fatalf("remove staged ASVS package: %v", err)
	}
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	result, err := pool.Exec(ctx, `
DELETE FROM artifact_bindings
 WHERE scope_kind = 'user' AND scope_id = $1
   AND namespace = $2 AND name = $3`,
		ownerID, auditstandards.CatalogNamespace,
		auditstandards.ArtifactName(auditstandards.Reference{Scheme: "owasp-asvs", Version: "5.0.0"}),
	)
	if err != nil {
		t.Fatalf("replace current ASVS catalog binding: %v", err)
	}
	if result.RowsAffected() != 1 {
		t.Fatalf("replaced ASVS catalog bindings = %d, want 1", result.RowsAffected())
	}
}

func assertASVSCatalogUnavailable(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	var page struct {
		Items []struct {
			Ref struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"ref"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audit-profiles?limit=100", &page)
	for _, profile := range page.Items {
		if profile.Ref.Name == "owasp-asvs-5-0-l1-source-review" && profile.Ref.Version == "1" {
			t.Fatalf("replaced ASVS profile remains current: %+v", profile)
		}
	}
	request, err := http.NewRequest(
		http.MethodGet, baseURL+"/v1/audit-standards/owasp-asvs/versions/5.0.0", nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusNotFound)
	response.Body.Close()
}

func assertASVSFindingBacktrace(
	t *testing.T,
	client *http.Client,
	baseURL, auditID string,
	fixture asvsBacktraceFixture,
) {
	t.Helper()
	var causal auditProgramFinding
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/findings/"+
		url.PathEscape(fixture.CausalFindingID), &causal)
	if causal.AnalystVerdict == nil || *causal.AnalystVerdict != "true_positive" ||
		causal.AnalystSeverity == nil || *causal.AnalystSeverity != "high" ||
		len(causal.FirstProposal.Document.StandardRefs) != 2 ||
		causal.FirstProposal.Origin.RunID != fixture.CausalRunID ||
		!causal.FirstProposal.Origin.RunDeleted || causal.FirstProposal.Origin.Audit == nil ||
		causal.FirstProposal.Origin.Audit.AuditID != auditID {
		t.Fatalf("rated ASVS finding was not retained exactly: %+v", causal)
	}
	var page struct {
		Items []struct {
			Kind   string `json:"kind"`
			Origin struct {
				RunDeleted bool   `json:"runDeleted"`
				RunID      string `json:"runId"`
				Audit      *struct {
					AuditID string `json:"auditId"`
				} `json:"audit,omitempty"`
			} `json:"origin"`
			Attempt *struct {
				RunDeleted    bool   `json:"runDeleted"`
				WorkflowRole  string `json:"workflowRole"`
				RunProvenance *struct {
					RunID    string `json:"runId"`
					Workflow *struct {
						Name    string `json:"name"`
						Version string `json:"version"`
					} `json:"workflow"`
				} `json:"runProvenance"`
				ItemOrigin struct {
					Standard *struct {
						Scheme           string   `json:"scheme"`
						Version          string   `json:"version"`
						MappingKey       string   `json:"mappingKey"`
						EntryIDs         []string `json:"entryIds"`
						EvidenceContract struct {
							ID      string `json:"id"`
							Version string `json:"version"`
						} `json:"evidenceContract"`
					} `json:"standard"`
				} `json:"itemOrigin"`
			} `json:"attempt"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/findings/"+
		url.PathEscape(fixture.CausalFindingID)+"/provenance?limit=100", &page)
	sawProposal, attemptCount := false, 0
	for _, record := range page.Items {
		switch record.Kind {
		case "source-proposal":
			sawProposal = record.Origin.RunDeleted && record.Origin.RunID == fixture.CausalRunID &&
				record.Origin.Audit != nil && record.Origin.Audit.AuditID == auditID
		case "check-attempt":
			origin := record.Attempt
			if origin == nil || !origin.RunDeleted || origin.RunProvenance == nil ||
				origin.RunProvenance.RunID != fixture.CausalRunID || origin.RunProvenance.Workflow == nil ||
				origin.RunProvenance.Workflow.Name != "audit-asvs-source-verification" ||
				origin.RunProvenance.Workflow.Version != "1" || origin.ItemOrigin.Standard == nil ||
				origin.ItemOrigin.Standard.Scheme != "owasp-asvs" ||
				origin.ItemOrigin.Standard.Version != "5.0.0" ||
				origin.ItemOrigin.Standard.MappingKey != "v5.0.0-1.2.4" ||
				!equalStrings(origin.ItemOrigin.Standard.EntryIDs, []string{"v5.0.0-1.2.4"}) ||
				origin.ItemOrigin.Standard.EvidenceContract.ID != "bounded-source-verification" ||
				origin.ItemOrigin.Standard.EvidenceContract.Version != "1" {
				t.Fatalf("ASVS check-attempt backtrace is incomplete: %+v", record)
			}
			attemptCount++
		}
	}
	if !sawProposal || attemptCount != 1 {
		t.Fatalf("ASVS provenance omitted deleted-Run records: %+v", page.Items)
	}

	var ordinary auditProgramFinding
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/findings/"+
		url.PathEscape(fixture.OrdinaryFindingID), &ordinary)
	if ordinary.AnalystVerdict != nil || ordinary.AnalystSeverity != nil ||
		ordinary.FirstProposal.ClientKey != "ordinary-asvs-mapping" ||
		ordinary.FirstProposal.Origin.RunID != fixture.OrdinaryRunID ||
		!ordinary.FirstProposal.Origin.RunDeleted || ordinary.FirstProposal.Origin.Audit != nil ||
		len(ordinary.FirstProposal.Document.StandardRefs) != 1 ||
		!hasASVSReference(ordinary, "v5.0.0-1.2.5") {
		t.Fatalf("ordinary ASVS-mapped finding gained a causal Audit origin: %+v", ordinary)
	}
	page.Items = nil
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/findings/"+
		url.PathEscape(fixture.OrdinaryFindingID)+"/provenance?limit=100", &page)
	if len(page.Items) != 1 || page.Items[0].Kind != "source-proposal" ||
		page.Items[0].Attempt != nil || !page.Items[0].Origin.RunDeleted ||
		page.Items[0].Origin.RunID != fixture.OrdinaryRunID || page.Items[0].Origin.Audit != nil {
		t.Fatalf("ordinary ASVS mapping was confused with a causal check attempt: %+v", page.Items)
	}
	var findingPage struct {
		Items []auditProgramFinding `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/findings?limit=100", &findingPage)
	if len(findingPage.Items) != 2 {
		t.Fatalf("ASVS mappings changed unique-finding cardinality: %+v", findingPage.Items)
	}
}

func hasASVSReference(finding auditProgramFinding, requirementID string) bool {
	for _, ref := range finding.FirstProposal.Document.StandardRefs {
		if ref.Scheme == "owasp-asvs" && ref.Version == "5.0.0" && ref.RequirementID == requirementID {
			return true
		}
	}
	return false
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func getAuditProgramCoverage(
	t *testing.T, client *http.Client, baseURL, auditID string,
) []auditProgramCoverage {
	t.Helper()
	var page struct {
		Items []auditProgramCoverage `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/coverage?limit=100", &page)
	return page.Items
}

func openAPIAuditBaselineGaps(
	t *testing.T, client *http.Client, baseURL, auditID string,
) []string {
	t.Helper()
	var audit struct {
		Baseline struct {
			Inventory struct {
				Gaps []string `json:"gaps"`
			} `json:"inventory"`
		} `json:"baseline"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID), &audit)
	return audit.Baseline.Inventory.Gaps
}

func assertAuditProgramReport(
	t *testing.T, client *http.Client, baseURL, auditID, conclusion string,
) {
	t.Helper()
	var report auditProgramReport
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/report", &report)
	if report.Status != "ready" || len(report.Machine) == 0 ||
		!strings.Contains(report.Summary, "Conclusion: "+conclusion) ||
		!strings.Contains(report.Summary, "not a security or compliance certification") {
		t.Fatalf("Audit report is not a truthful non-certifying report: %+v", report)
	}
}

func deleteCollectedAuditRuns(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, auditID string,
) {
	t.Helper()
	var page struct {
		Items []auditProgramItem `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/items?limit=100", &page)
	if len(page.Items) != 2 {
		t.Fatalf("Audit item count = %d, want 2", len(page.Items))
	}
	runIDs := make(map[string]struct{}, len(page.Items))
	for _, item := range page.Items {
		if item.State != "settled" || item.FinalDisposition != "accepted-result" ||
			item.AcceptedResult == nil || len(item.Attempts) != 1 ||
			item.Attempts[0].CollectionDisposition != "accepted-result" || item.Attempts[0].RunID == "" {
			t.Fatalf("Audit item was not durably collected: %+v", item)
		}
		runIDs[item.Attempts[0].RunID] = struct{}{}
	}
	for runID := range runIDs {
		waitForAuditRunDeletable(t, ctx, client, baseURL, runID)
		request, err := http.NewRequest(
			http.MethodDelete, baseURL+"/v1/runs/"+url.PathEscape(runID), nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response := do(t, client, request, http.StatusNoContent)
		response.Body.Close()
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/items?limit=100", &page)
	for _, item := range page.Items {
		if len(item.Attempts) != 1 || !item.Attempts[0].RunDeleted {
			t.Fatalf("deleted child Run lost its Audit tombstone: %+v", item)
		}
	}
}

func waitForAuditRunDeletable(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, runID string,
) {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		var run runStatus
		if auditProgramTryGET(ctx, client, baseURL+"/v1/runs/"+url.PathEscape(runID), &run) == nil && run.Deletable {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Audit child Run %s deletion: %v", runID, ctx.Err())
		case <-ticker.C:
		}
	}
}

func auditProgramGET(t *testing.T, client *http.Client, target string, output any) {
	t.Helper()
	if err := auditProgramTryGET(context.Background(), client, target, output); err != nil {
		t.Fatal(err)
	}
}

func decodeAuditProgramResponse(t *testing.T, response *http.Response, output any) {
	t.Helper()
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		t.Fatalf("decode HTTP %d response: %v", response.StatusCode, err)
	}
}

func auditProgramTryGET(ctx context.Context, client *http.Client, target string, output any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s returned HTTP %d", target, response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		return fmt.Errorf("decode GET %s: %w", target, err)
	}
	return nil
}

func auditProgramSourceArchive(t *testing.T) []byte {
	t.Helper()
	return auditProgramZip(t, map[string][]byte{
		"app.py": readAuditFixture(t, filepath.Join("source", "app.py")),
	})
}

func auditProgramZip(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write(files[name]); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}
