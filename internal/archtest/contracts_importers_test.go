package archtest

import (
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// contractsKernel is the private wire kernel. It holds the Server/Runtime
// contracts and the vocabulary they share: artifact refs, names and media
// types, identifier grammars, SecretString, validation errors and the strict
// codec. Domain rules live with their owners: Run Skill provenance in
// agentskills, the Stage result in planner/stageresult, Audit task contracts,
// the finding collection media type, Planner model policies and the
// audit-results Toolset rules in config/resolved, Runtime adapter and
// credential kinds in contracts/runtimesettings, and pinned selections in
// contentdigest.
const contractsKernel = "internal/contracts"

// contractsImporters lists every non-test package that imports the kernel,
// with kernel symbols it uses that the private wire defines. A new importer
// is added for such a symbol; a package that needs only a domain rule
// imports the rule's owner instead.
var contractsImporters = map[string][]string{
	"internal/agentprincipal":              {"ValidRuntimeAgentID"},
	"internal/agentskills":                 {"ArtifactRef", "ValidAgentSkillName"},
	"internal/app":                         {"ArtifactRef", "SecretString"},
	"internal/artifacts":                   {"ArtifactRef", "ValidateArtifactName", "ValidMediaType"},
	"internal/artifacts/artifactstest":     {"ArtifactRef"},
	"internal/auditcontroller":             {"ArtifactRef"},
	"internal/auditdomain":                 {"ArtifactRef", "ValidMediaType"},
	"internal/auditimport":                 {"ArtifactRef"},
	"internal/auditscan":                   {"ArtifactRef"},
	"internal/auditservice":                {"ArtifactRef", "ValidResourceID"},
	"internal/auditstandards":              {"ArtifactRef", "ValidConfigVersion"},
	"internal/auditstore":                  {"ArtifactRef"},
	"internal/auditstore/internal/pgstore": {"ArtifactReadResult", "ValidResourceID"},
	"internal/auth":                        {"ValidName"},
	"internal/cli":                         {"ValidateArtifactName"},
	"internal/config":                      {"ResolvedAgentTemplate", "ResolvedModelPolicy", "ToolsetSelection"},
	"internal/config/auditprofile":         {"AuditCheckResultsV1", "ValidConfigVersion"},
	"internal/config/resolved":             {"ResolvedAgentTemplate", "ResolvedModelPolicy", "ToolsetSelection"},
	"internal/contracts/control":           {"ResolvedAgentTemplate", "WorkerCompletionContract", "WorkerHandle"},
	"internal/contracts/llmgateway":        {"ValidConfigID", "ValidateDigest"},
	"internal/contracts/reporting":         {"DecodePrivateStrict", "ValidateWorkerSubtaskID"},
	"internal/contracts/runlabels":         {"Invalidf"},
	"internal/contracts/runtimesettings":   {"ResolvedModelPolicy", "SecretString"},
	"internal/contracts/scan":              {"ArtifactRef", "ValidateArtifactName"},
	"internal/controlplane":                {"DecodePrivateStrict", "ResolvedSkill", "WorkerHandle"},
	"internal/credentials":                 {"ResolvedModelPolicy", "SecretString"},
	"internal/credentials/litellm":         {"ResolvedModelPolicy"},
	"internal/evalservice":                 {"ArtifactRef"},
	"internal/evalstore":                   {"ArtifactRef", "ValidResourceID"},
	"internal/findingintake":               {"ArtifactRef", "ToolsetSelection"},
	"internal/httpapi/privateartifacts":    {"ArtifactReadResult", "ArtifactWriteResult", "ValidateArtifactListCursor"},
	"internal/httpapi/public":              {"ArtifactRef", "ValidRuntimeAgentID", "WorkerSessionMode"},
	"internal/httpapi/public/events":       {"ValidResourceID"},
	"internal/mtls":                        {"ValidRuntimeAgentID"},
	"internal/planner":                     {"StageContentRequest", "WorkerCompletion", "WorkerHandle"},
	"internal/planner/a2a":                 {"DecodePrivateStrict", "StageContentRequest", "WorkerCompletion"},
	"internal/planner/scan":                {"StageContentRequest", "WorkerCompletion"},
	"internal/planner/session":             {"ArtifactRef"},
	"internal/planner/stageresult":         {"ArtifactRef", "ResultJSONSize", "TerminationError"},
	"internal/planner/stateview":           {"WorkerCompletion", "WorkerHandle"},
	"internal/planner/streamline":          {"StageContentRequest", "WorkerResult"},
	"internal/projectstore":                {"ValidResourceID"},
	"internal/runrepeat":                   {"ArtifactRef", "ValidateArtifactName"},
	"internal/runservice":                  {"AgentSkillNamespace", "ArtifactRef"},
	"internal/runstore":                    {"ArtifactRef", "ValidRuntimeAgentID", "WorkerCompletionContract"},
	"internal/runtimeconfig":               {"ResolvedModelPolicy", "ValidIdentifier"},
	"internal/scanplan":                    {"ResolvedAgentTemplate", "ToolExecutionConfig"},
	"internal/scheduler":                   {"AllocationWorkspaceSpec", "ResolvedSkill", "WorkerHandle"},
	"internal/telemetry":                   {"DecodePrivateStrict", "ResultJSONSize"},
}

func TestContractsKernelImportersUseWireSymbols(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	module, err := modulePath(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	graph, err := loadImportGraph(root, module)
	if err != nil {
		t.Fatal(err)
	}
	used := map[string]map[string]bool{}
	for _, pkg := range sortedKeys(graph) {
		if _, imports := graph[pkg][contractsKernel]; !imports {
			continue
		}
		symbols, err := kernelSymbols(root, pkg, module+"/"+contractsKernel)
		if err != nil {
			t.Fatal(err)
		}
		used[pkg] = symbols
	}
	if len(used) == 0 {
		t.Fatal("found no importer of " + contractsKernel)
	}
	if problems := checkContractsImporters(contractsImporters, used); len(problems) > 0 {
		t.Fatal("contracts kernel importers changed; the list is contractsImporters in " +
			"internal/archtest/contracts_importers_test.go.\n\n" + strings.Join(problems, "\n"))
	}
}

// The check is tested on synthetic importers so each rule is known to fail.
func TestContractsImporterCheckReportsEachRule(t *testing.T) {
	listed := map[string][]string{
		"internal/kept":    {"ArtifactRef"},
		"internal/stale":   {"ArtifactRef"},
		"internal/drifted": {"ArtifactRef", "ValidName"},
		"internal/bare":    {},
	}
	used := map[string]map[string]bool{
		"internal/kept":    {"ArtifactRef": true, "ErrIdempotencyConflict": true},
		"internal/drifted": {"ArtifactRef": true},
		"internal/bare":    {"ArtifactRef": true},
		"internal/new":     {"ArtifactRef": true},
	}
	problems := checkContractsImporters(listed, used)
	for _, want := range []string{
		"internal/new imports internal/contracts but is not listed",
		"internal/stale is listed but no longer imports internal/contracts",
		"internal/drifted lists ValidName, which it no longer uses",
		"internal/bare lists no wire symbol",
	} {
		if !slices.ContainsFunc(problems, func(problem string) bool { return strings.Contains(problem, want) }) {
			t.Errorf("no problem reports %q; got:\n%s", want, strings.Join(problems, "\n"))
		}
	}
	for _, problem := range problems {
		if strings.Contains(problem, "internal/kept") {
			t.Errorf("a listed importer that uses its symbols was reported: %s", problem)
		}
	}
}

// checkContractsImporters holds the importers of the kernel, with the kernel
// symbols each uses, to the listed importers and their wire symbols.
func checkContractsImporters(listed map[string][]string, used map[string]map[string]bool) []string {
	var problems []string
	for _, pkg := range sortedKeys(used) {
		if _, ok := listed[pkg]; !ok {
			problems = append(problems, fmt.Sprintf(
				"%s imports %s but is not listed; list the wire symbols it uses, or import the owner of a domain rule instead",
				pkg, contractsKernel))
		}
	}
	for _, pkg := range sortedKeys(listed) {
		symbols, imports := used[pkg]
		switch {
		case !imports:
			problems = append(problems, fmt.Sprintf("%s is listed but no longer imports %s; delete the entry", pkg, contractsKernel))
		case len(listed[pkg]) == 0:
			problems = append(problems, fmt.Sprintf("%s lists no wire symbol", pkg))
		default:
			for _, symbol := range listed[pkg] {
				if !symbols[symbol] {
					problems = append(problems, fmt.Sprintf("%s lists %s, which it no longer uses", pkg, symbol))
				}
			}
		}
	}
	return problems
}

// kernelSymbols returns the exported kernel identifiers that the non-test Go
// files of pkg select through their import of kernelPath, counting the same
// files as loadImportGraph.
func kernelSymbols(root, pkg, kernelPath string) (map[string]bool, error) {
	entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(pkg)))
	if err != nil {
		return nil, err
	}
	symbols := map[string]bool{}
	fset := token.NewFileSet()
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, filepath.Join(root, filepath.FromSlash(pkg), name), nil, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		if ignoredFile(file) {
			continue
		}
		local := ""
		for _, spec := range file.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return nil, err
			}
			if path != kernelPath {
				continue
			}
			local = filepath.Base(kernelPath)
			if spec.Name != nil {
				local = spec.Name.Name
			}
		}
		if local == "" {
			continue
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if selector, ok := node.(*ast.SelectorExpr); ok {
				if identifier, ok := selector.X.(*ast.Ident); ok && identifier.Name == local {
					symbols[selector.Sel.Name] = true
				}
			}
			return true
		})
	}
	return symbols, nil
}

// ignoredFile reports a //go:build ignore file, which loadImportGraph skips.
func ignoredFile(file *ast.File) bool {
	for _, group := range file.Comments {
		if group.Pos() > file.Package {
			break
		}
		for _, comment := range group.List {
			if expression, err := constraint.Parse(comment.Text); err == nil {
				if tag, ok := expression.(*constraint.TagExpr); ok && tag.Tag == "ignore" {
					return true
				}
			}
		}
	}
	return false
}
