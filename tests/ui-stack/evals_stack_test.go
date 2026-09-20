//go:build e2e

package uistack

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (s *uiStack) runManagedEvals(owner, uiURL, apiURL, uiDirect, apiDirect, mode string) {
	t := s.t
	python := filepath.Join(s.repositoryRoot, "runtime", ".venv", "bin", "python")
	runChecked(t, s.repositoryRoot, cleanEnvironment(nil, "PATH"), python, "-c",
		"import importlib.util; assert importlib.util.find_spec('playground_evals') is None")
	producer := filepath.Join(s.repositoryRoot, "tests", "ui-stack", "evals_producer.py")
	producerEnv := cleanEnvironment(map[string]string{"EVAL_GATE_TOKEN": publicBearerCanary}, "PATH", "LANG")
	runChecked(t, s.repositoryRoot, producerEnv, "python3", "-I", "-S", producer,
		"seed", "--api", s.serverInternalURL, "--directory", s.temporaryRoot)
	output := filepath.Join(s.temporaryRoot, "eval-playwright-output")
	if evidence := os.Getenv("CONTRACTOR_EVAL_EVIDENCE_DIR"); evidence != "" {
		output = filepath.Join(evidence, "browser")
	}
	environment := cleanEnvironment(map[string]string{
		"CONTRACTOR_UI_EVAL_STACK":           "1",
		"CONTRACTOR_UI_E2E_BASE_URL":         uiURL,
		"CONTRACTOR_UI_E2E_API_URL":          apiURL,
		"CONTRACTOR_UI_E2E_API_DIRECT_URL":   apiDirect,
		"CONTRACTOR_UI_E2E_UI_DIRECT_URL":    uiDirect,
		"CONTRACTOR_UI_E2E_CONTROL_URL":      s.controlURL,
		"CONTRACTOR_UI_E2E_CONTROL_TOKEN":    s.controlToken,
		"CONTRACTOR_UI_E2E_USERNAME":         "admin",
		"CONTRACTOR_UI_E2E_PASSWORD":         uiStackPassword,
		"CONTRACTOR_EVAL_FIXTURE_DIR":        s.temporaryRoot,
		"CONTRACTOR_UI_E2E_OUTPUT_DIR":       output,
		"pnpm_config_verify_deps_before_run": "warn",
	}, "PATH", "HOME", "XDG_CACHE_HOME", "COREPACK_HOME", "TMPDIR", "LANG", "LC_ALL", "TZ")
	assertEnvironmentHasNoSecrets(t, environment, s.secrets)
	if mode == "external" {
		runChecked(t, s.repositoryRoot, producerEnv, "python3", "-I", "-S", producer,
			"external", "--api", s.serverInternalURL, "--directory", s.temporaryRoot)
		// The same UI reads producer results without a producer command route.
		environment = append(environment, "CONTRACTOR_EVAL_EXTERNAL_ONLY=1")
	}
	runChecked(t, filepath.Join(s.repositoryRoot, "ui"), environment,
		"corepack", "pnpm", "exec", "playwright", "test", "e2e/evals-stack.spec.ts", "--output", filepath.Join(output, mode))
	s.assertManagedEvalExecutions(owner, mode)
	s.assertManagedEvalOwnerBoundary(owner)
	_, _, failures := s.modelGateway.snapshot()
	if len(failures) != 0 {
		t.Fatalf("deterministic Evals Gateway failures: %v", failures)
	}
	if path := os.Getenv("CONTRACTOR_EVAL_EVIDENCE_DIR"); path != "" {
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}
		name := mode + "-evidence.json"
		raw, err := os.ReadFile(filepath.Join(s.temporaryRoot, name))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(raw), "PRIVATE_MANAGED_EVAL_RELEASE_TRUTH") {
			t.Fatal("private truth leaked into exported evidence")
		}
		if err = os.WriteFile(filepath.Join(path, name), raw, 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%s Workflow/Audit process journeys passed", mode)
}

func (s *uiStack) assertManagedEvalOwnerBoundary(owner string) {
	s.t.Helper()
	rows, err := s.pool.Query(s.ctx, `SELECT experiment_id, min(member_id)
FROM eval_experiments JOIN eval_members USING(experiment_id)
WHERE owner_id=$1 GROUP BY experiment_id`, owner)
	if err != nil {
		s.t.Fatal(err)
	}
	paths := []string{}
	for rows.Next() {
		var experiment, member string
		if err := rows.Scan(&experiment, &member); err != nil {
			s.t.Fatal(err)
		}
		base := "/v1/eval-experiments/" + experiment
		for _, suffix := range []string{"", "/members", "/pairs", "/report", "/members/" + member + "/review", "/members/" + member + "/executions"} {
			paths = append(paths, base+suffix)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		s.t.Fatal(err)
	}
	otherOwner := owner + "-other-owner"
	writeLocalAuth(s.t, s.temporaryRoot, otherOwner)
	if err := s.restartServer(); err != nil {
		s.t.Fatal(err)
	}
	client := &http.Client{Timeout: 5 * time.Second}
	for _, path := range paths {
		request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, s.serverInternalURL+path, nil)
		if err != nil {
			s.t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicBearerCanary)
		response, err := client.Do(request)
		if err != nil {
			s.t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 64*1024))
		response.Body.Close()
		if response.StatusCode != http.StatusNotFound {
			s.t.Fatalf("foreign-owner GET %s: %d, want 404", path, response.StatusCode)
		}
	}
	s.t.Logf("foreign owner cannot read %d experiment/evidence routes", len(paths))
}

func (s *uiStack) assertManagedEvalExecutions(owner, mode string) {
	s.t.Helper()
	rows, err := s.pool.Query(s.ctx, `
SELECT e.control_mode,min(sub.execution_kind),e.expected_count,count(DISTINCT m.member_id),count(DISTINCT sub.execution_id)
FROM eval_experiments e JOIN eval_members m USING(experiment_id)
LEFT JOIN eval_submissions sub USING(experiment_id,member_id)
WHERE e.owner_id=$1 GROUP BY e.experiment_id ORDER BY e.control_mode,min(sub.execution_kind)`, owner)
	if err != nil {
		s.t.Fatal(err)
	}
	defer rows.Close()
	var matrix []string
	expectedMode := "external"
	if mode == "native" {
		expectedMode = "server"
	}
	for rows.Next() {
		var controlMode, kind string
		var expected, members, executions int
		if err := rows.Scan(&controlMode, &kind, &expected, &members, &executions); err != nil {
			s.t.Fatal(err)
		}
		if controlMode != expectedMode || expected != 8 || members != expected || executions != expected {
			s.t.Fatalf("%s/%s matrix expected/members/ordinary executions = %d/%d/%d", controlMode, kind, expected, members, executions)
		}
		matrix = append(matrix, controlMode+"/"+kind)
	}
	if err = rows.Err(); err != nil {
		s.t.Fatal(err)
	}
	if len(matrix) != 2 || matrix[0] != expectedMode+"/audit" || matrix[1] != expectedMode+"/run" {
		s.t.Fatalf("managed matrix %v, want independent %s Workflow and Audit experiments", matrix, expectedMode)
	}
	raw, _ := json.Marshal(matrix)
	s.t.Log(fmt.Sprintf("ordinary execution membership: %s", raw))
}
