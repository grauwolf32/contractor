package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"go.yaml.in/yaml/v4"
)

type taskAuditManagement struct {
	fakeAuditManagement
	creation auditservice.TaskCreationParams
	list     auditstore.ListTasksParams
	task     auditstore.Task
	replayed bool
	calls    int
}

func (f *taskAuditManagement) CreateTasks(_ context.Context, p auditservice.TaskCreationParams) (auditservice.TaskCreationResult, error) {
	f.creation, f.calls = p, f.calls+1
	return auditservice.TaskCreationResult{Acceptance: auditstore.TaskAcceptance{TaskIDs: []string{f.task.Document.TaskID}, CreatedCount: 1, AuditRevision: 3}, Replayed: f.replayed}, nil
}

func (f *taskAuditManagement) Tasks(_ context.Context, p auditstore.ListTasksParams) (auditstore.TaskPage, error) {
	if !pagelimit.ValidPage(p.Limit) {
		return auditstore.TaskPage{}, auditstore.ErrInvalid
	}
	f.list = p
	page := auditstore.TaskPage{Tasks: []auditstore.Task{f.task}}
	if p.AfterOrdinal == 0 {
		ordinal := f.task.Ordinal
		page.NextOrdinal = &ordinal
	}
	return page, nil
}

func (f *taskAuditManagement) Task(_ context.Context, owner, audit, task string) (auditstore.Task, error) {
	if owner != "user-1" || audit != "audit-one" || task != f.task.Document.TaskID {
		return auditstore.Task{}, auditstore.ErrNotFound
	}
	return f.task, nil
}

func TestAuditTaskOperationsConformAndBindCursors(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	management := &taskAuditManagement{task: auditstore.Task{
		Document: auditcontract.Task{Schema: auditcontract.TaskSchema, AuditID: "audit-one", TaskID: "task-one", Workflow: "inspect", Inputs: map[string]auditcontract.Material{}, Parameters: map[string]string{}, Objective: "Inspect selected material"},
		Source:   auditstore.TaskSource{Kind: "manual", Key: "first"}, Ordinal: 7, Iteration: 1,
		State: auditstore.TaskReady, CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) { services.Audits = management })
	path := "/v1/audits/audit-one/tasks"
	body := `{"work":[{"key":"first","workflow":"inspect","objective":"Inspect selected material","inputs":{}}]}`
	for _, replayed := range []bool{false, true} {
		management.replayed = replayed
		response := contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, body, "Idempotency-Key", "accept-first", "If-Match", `"2"`), http.StatusOK)
		if response.Header().Get("ETag") != `"3"` || (response.Header().Get("Idempotency-Replayed") == "true") != replayed || management.creation.OwnerID != "user-1" || management.creation.ExpectedRevision != 2 || management.creation.AuditID != "audit-one" {
			t.Fatalf("lost task mutation authority or receipt: %+v %v", management.creation, response.Header())
		}
	}
	first := contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"?limit=1&state=ready", ""), http.StatusOK)
	var page struct{ Page pageInfoResponse }
	if json.Unmarshal(first.Body.Bytes(), &page) != nil || page.Page.NextCursor == nil {
		t.Fatal("missing bounded continuation")
	}
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"?limit=1&state=ready&cursor="+*page.Page.NextCursor, ""), http.StatusOK)
	if management.list.AfterOrdinal != 7 || management.list.Limit != 1 || management.list.OwnerID != "user-1" {
		t.Fatalf("lost bounded selection: %+v", management.list)
	}
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"?state=settled&cursor="+*page.Page.NextCursor, ""), http.StatusBadRequest)
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path+"/task-one", ""), http.StatusOK)
	for _, invalid := range []string{`{"work":[]}`, `{"work":[],"Work":[]}`, strings.Replace(body, `"workflow":"inspect"`, `"workflow":"inspect","workflow":"other"`, 1)} {
		before := management.calls
		contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodPost, path, invalid, "Idempotency-Key", "bad", "If-Match", `"2"`), http.StatusBadRequest)
		if management.calls != before {
			t.Fatal("invalid body reached Task owner")
		}
	}
}

// All common DTO rules retain their owning schema, including required fields,
// nested member grammar and metadata bounds. Explicit API intake additionally
// permits an empty input map so pinned Audit defaults can supply the inputs.
func TestSharedPublicAuditTaskSchemasMatchCommonContract(t *testing.T) {
	var common struct {
		Defs map[string]any `json:"$defs"`
	}
	var wire struct {
		Defs map[string]any `json:"$defs"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "v1alpha1", "common.schema.json"), &wire); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "audit", "v1", "contract.schema.json"), &common); err != nil {
		t.Fatal(err)
	}
	var public struct {
		Components struct{ Schemas map[string]any }
	}
	if err := yaml.Unmarshal(contractstest.ReadFile(t, "api", "openapi", "contractor-public-v1.yaml"), &public); err != nil {
		t.Fatal(err)
	}
	names := map[string]string{"key": "AuditCommonKey", "text": "AuditCommonText", "digest": "AuditCommonDigest", "resource_id": "AuditCommonResourceId", "ref": "AuditCommonExactRef", "artifact": "AuditTaskMaterial", "hypothesis_ref": "AuditTaskHypothesisRef", "task": "AuditTaskDocument", "prepared_selection": "AuditTaskSelection"}
	var convert func(any, bool) any
	convert = func(value any, external bool) any {
		switch v := value.(type) {
		case map[string]any:
			if ref, ok := v["$ref"].(string); ok {
				key := ref[strings.LastIndex(ref, "/")+1:]
				if external || !strings.HasPrefix(ref, "#/") {
					selected := wire.Defs[key]
					if key == "exactArtifactRef" {
						bytes, _ := json.Marshal(wire.Defs["artifactRef"])
						var flattened map[string]any
						_ = json.Unmarshal(bytes, &flattened)
						flattened["required"] = []string{"namespace", "name", "revision"}
						selected = flattened
					}
					if selected == nil {
						t.Fatalf("unknown common reference %s", ref)
					}
					return convert(selected, true)
				}
				name := names[key]
				if name == "" {
					t.Fatalf("unknown Audit reference %s", ref)
				}
				return map[string]any{"$ref": "#/components/schemas/" + name}
			}
		}
		v := value
		switch v := v.(type) {
		case []any:
			for index, x := range v {
				v[index] = convert(x, external)
			}
		case map[string]any:
			for key, x := range v {
				v[key] = convert(x, external)
			}
			if bound, exists := v["x-maxUtf8Bytes"]; exists {
				v["x-max-utf8-bytes"] = bound
				delete(v, "x-maxUtf8Bytes")
			}
			if v["const"] != nil || v["enum"] != nil {
				v["type"] = "string"
			}
		}
		return v
	}
	for key, name := range names {
		expected := convert(common.Defs[key], false)
		if key == "prepared_selection" {
			expected.(map[string]any)["properties"].(map[string]any)["inputs"].(map[string]any)["minProperties"] = 0
		}
		// Normalize YAML/JSON integer representation before comparison.
		canonical := func(value any) any {
			data, _ := json.Marshal(value)
			var normalized any
			_ = json.Unmarshal(data, &normalized)
			return normalized
		}
		if !reflect.DeepEqual(canonical(expected), canonical(public.Components.Schemas[name])) {
			t.Fatalf("public %s differs from common %s", name, key)
		}
	}
}

func TestAuditTaskGetRejectsQuery(t *testing.T) {
	h := auditTestHandler(&taskAuditManagement{})
	r := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/audit-one/tasks/task-one?unknown=1", nil)
	w := httptest.NewRecorder()
	serve(h.getAuditTask).ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unexpected query: %d %s", w.Code, w.Body.String())
	}
}
