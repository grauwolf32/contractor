package public

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
)

// requireOperationCoverageEnv asks a filtered run to enforce the operation
// coverage too; make verify-public-api sets it for its contract tests.
const requireOperationCoverageEnv = "CONTRACTOR_REQUIRE_OPERATION_COVERAGE"

// TestMain enforces that every operation of the public contract passed
// serveAndValidatePublicContract, after a passing run of the whole package
// or of a filtered run that sets CONTRACTOR_REQUIRE_OPERATION_COVERAGE=1.
// Every operation needs a validated success response, with two exceptions
// that need one validated response of any status: the WebSocket upgrade,
// whose 101 needs a hijackable connection that the recorder does not offer
// (the event handler tests complete the handshake on a real server), and,
// without CONTRACTOR_TEST_DATABASE_URL, the Eval operations, whose success
// responses come from the Eval service on PostgreSQL.
func TestMain(m *testing.M) {
	code := m.Run()
	if code == 0 && operationCoverageRequired() {
		document, err := openapi3.NewLoader().LoadFromFile(publicOpenAPIPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load public OpenAPI for the operation coverage: %v\n", err)
			os.Exit(1)
		}
		withPostgreSQL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL") != ""
		if missing := publicContractLedger.uncovered(document, withPostgreSQL); len(missing) != 0 {
			fmt.Fprintf(os.Stderr, "public operations without a validated test response:\n  %s\n",
				strings.Join(missing, "\n  "))
			code = 1
		}
	}
	os.Exit(code)
}

// operationCoverageRequired reports a run of every test of the package, or
// a run that asks for the coverage.
func operationCoverageRequired() bool {
	if os.Getenv(requireOperationCoverageEnv) == "1" {
		return true
	}
	for _, name := range []string{"test.run", "test.skip", "test.list"} {
		if selected := flag.Lookup(name); selected != nil && selected.Value.String() != "" {
			return false
		}
	}
	return true
}

// contractLedger records the operations whose responses a test validated
// against the OpenAPI contract: present when any response was validated,
// true when a success response was.
type contractLedger struct {
	mu        sync.Mutex
	validated map[string]bool
}

var publicContractLedger = &contractLedger{validated: map[string]bool{}}

func (l *contractLedger) record(operationID string, status int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.validated[operationID] = l.validated[operationID] || status < http.StatusMultipleChoices
}

// uncovered lists the operations of document that lack the validated
// response TestMain requires.
func (l *contractLedger) uncovered(document *openapi3.T, withPostgreSQL bool) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	var missing []string
	for path, item := range document.Paths.Map() {
		for method, operation := range item.Operations() {
			success, validated := l.validated[operation.OperationID]
			anyStatus := operation.OperationID == "connectEventsWebSocket" ||
				!withPostgreSQL && slices.Contains(operation.Tags, "Evals")
			switch {
			case !validated:
				missing = append(missing, fmt.Sprintf("%s %s (%s): no validated response", method, path, operation.OperationID))
			case !success && !anyStatus:
				missing = append(missing, fmt.Sprintf("%s %s (%s): no validated success response", method, path, operation.OperationID))
			}
		}
	}
	sort.Strings(missing)
	return missing
}

// TestRouteTableMatchesOpenAPI keeps the one route table equal to the
// operations of the public contract: every operation has exactly one route
// with its operationId, and no route serves an undocumented operation.
func TestRouteTableMatchesOpenAPI(t *testing.T) {
	document := loadPublicOpenAPI(t)
	documented := map[string]string{}
	for path, item := range document.Paths.Map() {
		for method, operation := range item.Operations() {
			documented[strings.ToUpper(method)+" "+path] = operation.OperationID
		}
	}
	served := map[string]string{}
	operationIDs := map[string]string{}
	for _, route := range (contextHandlers{}).routes() {
		key := route.method + " " + route.path
		if previous, ok := served[key]; ok {
			t.Errorf("%s is routed twice, as %s and %s", key, previous, route.operationID)
		}
		if previous, ok := operationIDs[route.operationID]; ok {
			t.Errorf("operationId %s is routed for %s and %s", route.operationID, previous, key)
		}
		if route.handler == nil {
			t.Errorf("%s has no handler", key)
		}
		served[key], operationIDs[route.operationID] = route.operationID, key
	}
	for key, operationID := range documented {
		switch routed, ok := served[key]; {
		case !ok:
			t.Errorf("OpenAPI operation %s (%s) has no route", key, operationID)
		case routed != operationID:
			t.Errorf("%s is routed as %s, but OpenAPI names it %s", key, routed, operationID)
		}
	}
	for key, operationID := range served {
		if _, ok := documented[key]; !ok {
			t.Errorf("route %s (%s) is not an OpenAPI operation", key, operationID)
		}
	}
	if len(served) == 0 {
		t.Fatal("the route table is empty")
	}
}
