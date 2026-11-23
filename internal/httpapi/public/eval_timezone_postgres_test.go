package public

import (
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestSharedEvalPostgresExperimentSummaryUTC(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for _, zone := range []string{"UTC", "Asia/Kolkata", "America/New_York"} {
		t.Run(zone, func(t *testing.T) {
			parsed, err := url.Parse(databaseURL)
			if err != nil {
				t.Fatal(err)
			}
			query := parsed.Query()
			query.Set("timezone", zone)
			parsed.RawQuery = query.Encode()
			t.Setenv("CONTRACTOR_TEST_DATABASE_URL", parsed.String())
			h := newEvalAPIHarness(t)
			draft, _ := h.dataset(t, "workflow")
			created := apiDecode[struct {
				ID string `json:"experimentId"`
			}](t, h.request(t, "POST", "/v1/projects/evaluation/eval-experiments",
				evaldomain.CreateExperiment{Name: "UTC summary", ControlMode: "server", Draft: &draft}, "utc-summary", "", 201))
			h.request(t, "GET", "/v1/eval-experiments/"+created.ID, nil, "", "", 200)
			var expected time.Time
			if err := h.pool.QueryRow(t.Context(), `SELECT updated_at FROM eval_experiments WHERE experiment_id=$1`, created.ID).Scan(&expected); err != nil {
				t.Fatal(err)
			}
			for _, path := range []string{"/v1/eval-experiments", "/v1/eval-experiments?projectId=evaluation"} {
				page := apiDecode[struct {
					Items []struct {
						UpdatedAt string `json:"updatedAt"`
					} `json:"items"`
				}](t, h.request(t, "GET", path, nil, "", "", 200))
				if len(page.Items) != 1 || !strings.HasSuffix(page.Items[0].UpdatedAt, "Z") {
					t.Fatalf("%s: expected one UTC summary, got %+v", path, page.Items)
				}
				actual, err := time.Parse(time.RFC3339Nano, page.Items[0].UpdatedAt)
				if err != nil || !actual.Equal(expected) {
					t.Fatalf("%s: summary timestamp changed its instant: %v", path, err)
				}
			}
		})
	}
}
