//go:build e2e

package e2e

import (
	"context"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Configuration parsing alone cannot detect profiles referring to standards
// removed from the Artifact catalog. Exercise actual Server startup and pins.
func TestAuditProgramCatalogReplacementRestartsServer(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot, temporaryRoot := repoRoot(t), t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := sharedServerBinary(t)
	runChecked(t, repositoryRoot, map[string]string{"CONTRACTOR_DATABASE_URL": isolateURL}, serverBinary, "migrate")

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	identity, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}},
		URI:         "urn:contractor:control-plane:audit-catalog-restart",
	})
	if err != nil {
		t.Fatal(err)
	}
	configRoot := stageE2EConfiguration(t, filepath.Join(repositoryRoot, "configs"), filepath.Join(temporaryRoot, "configs"), "http://127.0.0.1:1/v1")
	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	baseURL := "http://" + publicAddress
	ownerID := "audit-catalog-owner-" + randomHex(t, 8)
	environment := map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_OPERATOR_CONFIG_ROOT":    configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             "https://" + privateAddress,
		"CONTRACTOR_CA_FILE":                 ca.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": identity.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  identity.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         writeE2ELocalAuth(t, temporaryRoot, ownerID),
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}
	client := &http.Client{Timeout: 8 * time.Second}
	server := startProcess(t, "Go Server before catalog replacement", repositoryRoot, environment, serverBinary, "serve")
	waitForHTTP(t, ctx, server, client, baseURL+"/readyz", http.StatusOK)
	createProjectResource(t, client, baseURL)
	var profiles struct {
		Items []struct {
			Ref struct {
				Name string `json:"name"`
			} `json:"ref"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audit-profiles?limit=100", &profiles)
	found := false
	for _, profile := range profiles.Items {
		found = found || profile.Ref.Name == "openapi-operation-trace"
	}
	if !found {
		t.Fatal("catalog replacement fixture did not start with the profile it removes")
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	replaceAuditProgramCatalog(t, ctx, isolateURL, ownerID, configRoot)
	server.stop(t)
	restarted := startProcess(t, "Go Server after catalog replacement", repositoryRoot, environment, serverBinary, "serve")
	waitForHTTP(t, ctx, restarted, client, baseURL+"/readyz", http.StatusOK)
	assertAuditProgramCatalogUnavailable(t, client, baseURL)
	t.Log("Server restarted without the removed current catalog sources")
}
