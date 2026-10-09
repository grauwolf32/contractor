package artifacts_test

import (
	"strings"
	"testing"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresMediaTypeLengthMatchesPublicationBound(t *testing.T) {
	pool := isolatedArtifactPool(t, t.Context())
	if _, err := pool.Exec(t.Context(), `INSERT INTO artifact_blobs (sha256,payload,size_bytes)
VALUES (sha256('fixture'::bytea),'fixture'::bytea,7)`); err != nil {
		t.Fatal(err)
	}
	for _, length := range []int{255, 256} {
		_, err := pool.Exec(t.Context(), `INSERT INTO artifact_versions(version_id,blob_sha256,media_type)
VALUES ($1,sha256('fixture'::bytea),$2)`, strings.Repeat("v", length), "text/"+strings.Repeat("a", length-5))
		if length == 255 && err != nil || length == 256 && postgres.SQLState(err) != "23514" {
			t.Fatalf("media type length %d: %v", length, err)
		}
	}
}
