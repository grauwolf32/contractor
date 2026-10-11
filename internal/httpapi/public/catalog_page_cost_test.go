package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configtest"
)

// TestWorkflowCatalogPageDoesNotCopyOrEncodeTheCatalog is the regression for
// GET /v1/workflows cloning, encoding and hashing the whole catalog on every
// page to fingerprint its cursor. The snapshot fingerprints its Workflows
// once, so one page of a catalog that encodes to about 2 MB allocates far
// less than that encoding (CAT-R67).
func TestWorkflowCatalogPageDoesNotCopyOrEncodeTheCatalog(t *testing.T) {
	root := configtest.CopyWithPolicies(t, "../../../testdata/configs")
	manifest, err := os.ReadFile(filepath.Join(root, "workflows", "artifact_copy.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const workflowCount, instructionBytes = 20, 100_000
	for index := range workflowCount {
		name := fmt.Sprintf("catalog-cost-%02d", index)
		copied := strings.Replace(string(manifest), "name: artifact-copy\n", "name: "+name+"\n", 1)
		copied = strings.Replace(copied, "ref: instructions/copy-planner.md", "ref: instructions/"+name+".md", 1)
		if copied == string(manifest) || !strings.Contains(copied, "instructions/"+name+".md") {
			t.Fatal("the Workflow manifest fixture changed shape")
		}
		writes := map[string]string{
			filepath.Join(root, "workflows", name+".yaml"):  copied,
			filepath.Join(root, "instructions", name+".md"): strings.Repeat("a", instructionBytes),
		}
		for path, content := range writes {
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	fixture := newHandlerFixtureWithConfig(t, root)
	encoded, err := json.Marshal(fixture.configs.Workflows())
	if err != nil {
		t.Fatal(err)
	}
	catalogBytes := uint64(len(encoded))
	if catalogBytes < workflowCount*instructionBytes {
		t.Fatalf("catalog encodes to %d bytes; the fixture must be realistic", catalogBytes)
	}

	page := func() {
		response := serveQuery(t, fixture.handler, "/v1/workflows?limit=1")
		var decoded workflowPageResponse
		decodeQueryResponse(t, response, &decoded)
		if response.Code != http.StatusOK || len(decoded.Items) != 1 || decoded.Page.NextCursor == nil {
			t.Fatalf("Workflow page = %d %+v", response.Code, decoded)
		}
	}
	page()
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	page()
	runtime.ReadMemStats(&after)
	allocated := after.TotalAlloc - before.TotalAlloc
	if allocated > catalogBytes/4 {
		t.Fatalf("one Workflow page allocated %d bytes for a catalog that encodes to %d bytes", allocated, catalogBytes)
	}
	t.Logf("one Workflow page allocated %d bytes; the catalog encodes to %d bytes", allocated, catalogBytes)
}
