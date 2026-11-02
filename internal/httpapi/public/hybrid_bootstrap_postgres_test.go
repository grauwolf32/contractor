package public

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentprincipal"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

// Run the actual Python configure/assign clients against the HTTP handlers and
// real PostgreSQL receipts. Only unrelated Gateway credential provisioning uses
// the existing test fixture; RuntimeConfig versions, bindings and principals are real.
func TestPostgresHybridBootstrapMutationIdentity(t *testing.T) {
	for _, scenario := range []string{"binding", "principal"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
			defer cancel()
			pool := isolatedPublicPool(t, ctx)
			state := t.TempDir()
			principalID := hybridBootstrapCertificate(t, state)
			registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{})
			if err != nil {
				t.Fatal(err)
			}
			principalService, err := agentprincipal.NewService(agentprincipal.ServiceOptions{Pool: pool, DeletionGuard: registry})
			if err != nil {
				t.Fatal(err)
			}
			principals, err := controlplane.NewPrincipalOperations(principalService, registry)
			if err != nil {
				t.Fatal(err)
			}
			fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(dependencies *testServices) {
				lookup := runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return dependencies.Credentials, nil })
				publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
					Pool: pool, RuntimeCredentials: hybridBootstrapCredentials{}, TransactionLLMCredentials: lookup,
					GatewayResolver: runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (llmgateway.ResolvedLLMGatewayConfig, error) {
						return dependencies.Config.(*config.Manager).LLMGateway(selector)
					}),
					PlannerTelemetryAdapters: runtimeconfigtest.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
				})
				if err != nil {
					t.Fatal(err)
				}
				bindings, err := runtimeconfig.NewBindingService(hybridBootstrapCredentials{}, lookup)
				if err != nil {
					t.Fatal(err)
				}
				management, err := runtimeconfig.NewManagementService(pool, publisher, bindings, agentprincipal.LabelHolders{})
				if err != nil {
					t.Fatal(err)
				}
				for _, version := range []string{"1", "2"} {
					document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"cloud-llm","version":"` + version + `"},"spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1","credential":"development-worker"}}}}`)
					published, err := publisher.Publish(ctx, document, "publish-cloud-"+version, "operator")
					if err != nil {
						t.Fatal(err)
					}
					if version == "1" {
						if _, err := management.CreateBinding(ctx, "unrelated", published.Version.Ref, "unrelated-label", "operator", time.Now()); err != nil {
							t.Fatal(err)
						}
					}
				}
				dependencies.RuntimeConfigs, dependencies.RuntimeAgentPrincipals = management, principals
				gateway, err := dependencies.Config.(*config.Manager).LLMGateway("local-litellm@1")
				if err != nil {
					t.Fatal(err)
				}
				encoded, err := json.Marshal(gateway.Ref)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(state, "gateway-ref.json"), encoded, 0600); err != nil {
					t.Fatal(err)
				}
			})
			if _, err := principalService.Register(ctx, principalID); err != nil {
				t.Fatal(err)
			}
			if _, err := principalService.ReplaceLabelsIdempotent(ctx, principalID, 1,
				[]string{"unrelated"}, "assign-unrelated", "operator", time.Now()); err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("X-Test-Lose-Response") == "yes" {
					response := httptest.NewRecorder()
					fixture.handler.ServeHTTP(response, r)
					if response.Code < 200 || response.Code >= 300 {
						t.Errorf("lost-response mutation did not commit: %d %s", response.Code, response.Body.String())
					}
					connection, _, err := w.(http.Hijacker).Hijack()
					if err != nil {
						t.Error(err)
						return
					}
					_ = connection.Close()
					return
				}
				fixture.handler.ServeHTTP(w, r)
			}))
			defer server.Close()
			token := filepath.Join(state, "api-token")
			if err := os.WriteFile(token, []byte(testBearerToken), 0600); err != nil {
				t.Fatal(err)
			}
			// The bootstrap helper is stdlib-only, like its documented python3 invocation.
			command := exec.CommandContext(ctx, "python3", "testdata/hybrid_bootstrap.py", server.URL, state, scenario, principalID)
			output, err := command.CombinedOutput()
			if err != nil {
				t.Fatalf("Python bootstrap %s: %v\n%s", scenario, err, output)
			}
			t.Logf("%s", output)
		})
	}
}

type hybridBootstrapCredentials struct{ parityRuntimeCredentials }

func (catalog hybridBootstrapCredentials) ForRuntimeTransaction(pgx.Tx) (runtimeconfig.RuntimeCredentialValidator, error) {
	return catalog, nil
}

func hybridBootstrapCertificate(t *testing.T, state string) string {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(1)}
	certificate, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	directory := filepath.Join(state, "agents")
	if err := os.Mkdir(directory, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "fixture-1.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate}), 0600); err != nil {
		t.Fatal(err)
	}
	publicKey, err := x509.MarshalPKIXPublicKey(&key.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(publicKey)
	return hex.EncodeToString(digest[:])
}
