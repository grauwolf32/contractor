package app

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestBlobConfigFlagsOverrideEnvironment(t *testing.T) {
	env := func(k string) string {
		switch k {
		case "CONTRACTOR_ARTIFACT_BLOB_BACKEND":
			return "s3"
		case "CONTRACTOR_ARTIFACT_BLOB_PATH":
			return "/unused"
		}
		return ""
	}
	cfg, err := ParseConfig([]string{"--artifact-blob-backend=postgresql", "--artifact-blob-path="}, env)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ArtifactBlobBackend != blob.Postgres || cfg.ArtifactBlobPath != "" {
		t.Fatal("flags did not win")
	}
	if _, err := ParseConfig(nil, env); err == nil {
		t.Fatal("S3 accepted")
	}
}
