package projectworkflows

import (
	"archive/zip"
	"bytes"
	"io"
	"testing"
)

func TestSourceFixtureIsSmallAndContainsNoGeneratedSpecifications(t *testing.T) {
	data, err := SourceArchive()
	if err != nil {
		t.Fatal(err)
	}
	if len(data) > 64<<10 {
		t.Fatalf("fixture ZIP is %d bytes", len(data))
	}
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{}
	for _, file := range reader.File {
		stream, openErr := file.Open()
		if openErr != nil {
			t.Fatal(openErr)
		}
		content, readErr := io.ReadAll(stream)
		stream.Close()
		if readErr != nil {
			t.Fatal(readErr)
		}
		files[file.Name] = string(content)
	}
	if len(files) != 3 || files["app.py"] != ApplicationSource || files["pyproject.toml"] != ProjectManifest {
		t.Fatalf("fixture entries = %v", files)
	}
	for name := range files {
		if name == "openapi.yaml" || name == "architecture.c4" {
			t.Fatalf("fixture contains generated specification %q", name)
		}
	}
}

func TestSemanticScorersAcceptCompleteOutputs(t *testing.T) {
	dependency := []byte("fastapi pyproject.toml:5\nasyncpg pyproject.toml:6\nhttpx app.py:4\n")
	project := []byte("GET /widgets/{widget_id} app.py:24\nPOST /widgets app.py:45\nPostgreSQL app.py:26\nInventory app.py:34\nBearer app.py:17\n")
	openAPI := []byte(`openapi: 3.0.3
info: {title: Widget Service, version: 1.0.0}
security: [{bearerAuth: []}]
paths:
  /widgets:
    x-path-files: [app.py]
    post:
      responses: {"201": {description: Created}}
  /widgets/{widget_id}:
    x-path-files: [app.py]
    get:
      responses: {"200": {description: Found}, "404": {description: Missing}}
components:
  securitySchemes:
    bearerAuth: {type: http, scheme: bearer}
`)
	likeC4 := []byte(`specification {
  element actor
  element system
  element container
  element database
  element external
}
model {
  client = actor 'API Client' { description 'Evidence app.py:17' }
  app = system 'Widget Service' {
    api = container 'API' { description 'Evidence app.py:24-55' }
    db = database 'PostgreSQL' { description 'Evidence app.py:26-51' }
  }
  inventory = external 'Inventory Service'
  client -> app.api 'Calls API'
  app.api -> app.db 'Persists widgets'
  app.api -> inventory 'Checks availability'
}
views { view index { include * } }
`)
	score := MergeScores(
		ScoreDiscovery(dependency, project), ScoreOpenAPI(openAPI), ScoreLikeC4(likeC4),
	)
	if !score.Passed() {
		t.Fatalf("complete fixtures failed: %+v", score.Failures)
	}
}

func TestSemanticScorersReturnPredicateCodesWithoutArtifactContent(t *testing.T) {
	secret := "fixture-secret-must-not-leak"
	score := MergeScores(
		ScoreDiscovery([]byte(secret), []byte(secret)),
		ScoreOpenAPI([]byte("not: [valid"+secret)),
		ScoreLikeC4([]byte(secret)),
	)
	if score.Passed() {
		t.Fatal("invalid fixtures unexpectedly passed")
	}
	seen := map[string]bool{}
	for _, failure := range score.Failures {
		if failure.Code == "" || failure.Message == "" {
			t.Fatalf("unbounded failure shape: %+v", failure)
		}
		if bytes.Contains([]byte(failure.Message), []byte(secret)) {
			t.Fatalf("failure leaked artifact content: %+v", failure)
		}
		seen[failure.Code] = true
	}
	for _, code := range []string{"openapi.parse", "likec4.application", "project.evidence"} {
		if !seen[code] {
			t.Fatalf("missing predicate %q in %+v", code, score.Failures)
		}
	}
}
