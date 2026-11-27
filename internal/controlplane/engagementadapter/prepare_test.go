package engagementadapter

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"net"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/mtlstest"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

type grantReader struct{ grant controlplane.AllocationGrant }

func (r *grantReader) GetGrant(string) (controlplane.AllocationGrant, error) { return r.grant, nil }

// Real mTLS protocol calls exercise the adapter/core seam, authority recheck,
// completed wire validation and secret scan together. No target I/O is involved.
func TestSharedEngagementPreparationPreservesAuthorityAndPrivateMaterialLifetime(t *testing.T) {
	for _, mode := range []string{"ordinary", "metadata", "controlled", "fenced", "fenced-after-material", "released", "storage-failure", "provider-failure", "cancelled", "agent-card-secret", "transport-failure"} {
		t.Run(mode, func(t *testing.T) {
			fixture, err := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "engagement-allocation-preflight.json"))
			if err != nil {
				t.Fatal(err)
			}
			projection := fixture.Engagement
			if mode == "ordinary" {
				projection = nil
			}
			if mode == "metadata" {
				metadata, decodeErr := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "allocation-spec-engagement-metadata.json"))
				if decodeErr != nil {
					t.Fatal(decodeErr)
				}
				projection = metadata.Engagement
			}
			ref := runtimesettings.RuntimeCredentialRef{CredentialID: "held-target", Kind: runtimesettings.RuntimeCredentialTargetBearer}
			if projection != nil && projection.HTTP != nil {
				projection.HTTP.Identities = append(projection.HTTP.Identities, engagementcontract.Identity{ID: "account-a", Kind: "bearer", Origins: []string{"https://target.example"}, Credential: &ref})
			}
			material := map[string]runtimesettings.HTTPIdentityMaterial{"account-a": {Credential: ref, Fields: map[string]contracts.SecretString{"token": contracts.NewSecretString("private-target-canary")}}}
			var received atomic.Int32
			var endpoint string
			files, principal, server := startRuntimeServer(t, func(ctx context.Context, wire *pb.PrepareAllocationRequest) (*pb.PrepareAllocationResponse, error) {
				received.Add(1)
				if mode == "transport-failure" {
					return nil, runtimewire.Error(ctx, codes.Unavailable, "unavailable", true)
				}
				var request control.PrepareAllocationRequest
				if err := runtimewire.Decode(wire, &request); err != nil {
					return nil, err
				}
				if !reflect.DeepEqual(request.Spec.Engagement, projection) {
					t.Error("prepare changed trusted projection or RPC")
				}
				if mode == "controlled" || mode == "agent-card-secret" {
					if request.Spec.RuntimeSettings.HTTPIdentities["account-a"].Fields["token"].Reveal() != "private-target-canary" {
						t.Error("assigned private material missing from prepare")
					}
				}
				public, _ := json.Marshal(request.Spec.Engagement)
				if strings.Contains(string(public), "private-target-canary") {
					t.Error("private material entered attribution")
				}
				card := agentCard(endpoint, request.Spec.AllocationID)
				if mode == "agent-card-secret" {
					card["description"] = "private-target-canary"
				}
				response := &pb.PrepareAllocationResponse{}
				err := runtimewire.Encode(control.PrepareAllocationResponse{APIVersion: contracts.APIVersion, WorkerHandle: contracts.WorkerHandle{AllocationID: request.Spec.AllocationID, AgentTemplateRef: request.Spec.AgentTemplate.Ref, WorkerRuntimeRef: request.Spec.AgentTemplate.Runtime, LeaseExpiresAt: request.Spec.LeaseExpiresAt, AgentCard: card}}, response)
				return response, err
			})
			endpoint, err = mtls.GRPCAuthority(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			reservation := controlplane.Reservation{
				Grant:      controlplane.AllocationGrant{AllocationID: fixture.AllocationID, RunID: fixture.RunID, StageExecutionID: fixture.StageExecutionID, LogicalAgentName: fixture.LogicalAgentName, Namespace: fixture.Namespace, RuntimeAgentID: principal, RuntimeInstanceID: "instance"},
				ControlURL: server.URL, A2AURL: server.URL, AgentTemplate: fixture.AgentTemplate, WorkerSessionMode: fixture.WorkerSessionMode, ContextManagement: fixture.ContextManagement, LeaseExpiresAt: fixture.LeaseExpiresAt,
			}
			grants := &grantReader{grant: reservation.Grant}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			materialCalls := 0
			adapter := &Adapter{
				projection: func(_ context.Context, selected controlplane.Reservation, _ runtimesettings.RuntimeSettings, _ bool) (*engagementcontract.AllocationProjection, error) {
					if !reflect.DeepEqual(selected.Grant, reservation.Grant) {
						t.Fatal("foreign reservation selected")
					}
					switch mode {
					case "fenced":
						grants.grant.WriteFenced = true
					case "released":
						grants.grant.RuntimeInstanceID = "replacement"
					case "storage-failure":
						return nil, errors.New("storage unavailable")
					}
					return projection, nil
				},
				materials: func(ctx context.Context, selected *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error) {
					materialCalls++
					if mode == "fenced-after-material" {
						grants.grant.WriteFenced = true
					}
					if selected != projection {
						t.Fatal("foreign identity assignment")
					}
					if _, ok := ctx.Deadline(); !ok {
						t.Fatal("unbounded identity lookup")
					}
					if mode == "provider-failure" {
						return material, errors.New("provider unavailable")
					}
					if mode == "cancelled" {
						cancel()
					}
					return material, nil
				},
			}
			client, err := controlplane.NewMTLSRuntimeControlClient(files, time.Second, grants, adapter)
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			_, err = client.Prepare(ctx, reservation, runtimesettings.WorkerExecutionSettings{ModelPolicy: fixture.ModelPolicy, RuntimeSettings: fixture.RuntimeSettings, ResolvedRuntimeConfigProvenance: fixture.ResolvedRuntimeConfigProvenance})
			success := mode == "ordinary" || mode == "metadata" || mode == "controlled"
			if (err == nil) != success || err != nil && strings.Contains(err.Error(), "private-target-canary") {
				t.Fatalf("unsafe prepare result: %v", err)
			}
			if materialCalls != 0 && len(material) != 0 {
				t.Fatal("private material retained after prepare")
			}
			if (received.Load() == 1) != (success || mode == "agent-card-secret" || mode == "transport-failure") {
				t.Fatalf("unexpected prepare dispatch: %d", received.Load())
			}
			if (mode == "ordinary" || mode == "metadata") && materialCalls != 0 {
				t.Fatal("metadata decrypted identities")
			}
		})
	}
}

func startRuntimeServer(t *testing.T, handler func(context.Context, *pb.PrepareAllocationRequest) (*pb.PrepareAllocationResponse, error)) (mtls.Files, string, *httptest.Server) {
	t.Helper()
	root := filepath.Join(t.TempDir(), "pki")
	generator := localpki.Generator{}
	ca, err := generator.InitCA(root, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	cp, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := generator.IssueAgent(root, "adapter-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	serverTLS, err := mtlstest.AgentServerConfig(mtlstest.Files{Certificate: agent.Certificate, PrivateKey: agent.PrivateKey, CA: ca.Certificate})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := os.ReadFile(agent.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(encoded)
	if block == nil {
		t.Fatal("missing agent certificate")
	}
	certificate, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(certificate)
	if err != nil {
		t.Fatal(err)
	}
	rpc := grpc.NewServer()
	pb.RegisterRuntimeLifecycleServer(rpc, &prepareRPCFixture{prepare: handler})
	t.Cleanup(rpc.Stop)
	server := httptest.NewUnstartedServer(rpc)
	server.EnableHTTP2 = true
	server.TLS = serverTLS
	server.StartTLS()
	t.Cleanup(server.Close)
	return mtls.Files{Certificate: cp.Certificate, PrivateKey: cp.PrivateKey, CA: ca.Certificate}, principal, server
}

func agentCard(endpoint, allocationID string) map[string]any {
	const input = "application/vnd.contractor.stage-content+json"
	const output = "application/vnd.contractor.worker-completion+json"
	return map[string]any{
		"name": "builder", "supportedInterfaces": []any{map[string]any{"url": endpoint, "protocolBinding": "GRPC", "protocolVersion": "1.0", "tenant": allocationID}},
		"defaultInputModes": []any{input}, "defaultOutputModes": []any{output},
		"skills":               []any{map[string]any{"id": "contractor_stage_content", "inputModes": []any{input}, "outputModes": []any{output}}},
		"securitySchemes":      map[string]any{"mutualTLS": map[string]any{"mtlsSecurityScheme": map[string]any{}}},
		"securityRequirements": []any{map[string]any{"schemes": map[string]any{"mutualTLS": map[string]any{}}}},
	}
}

func TestSharedEngagementRejectsUncontrolledNetworkBeforeIdentityLookup(t *testing.T) {
	fixture, err := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "engagement-allocation-preflight.json"))
	if err != nil {
		t.Fatal(err)
	}
	fixture.AgentTemplate.SandboxProfile.SandboxProfileID = "podman-network"
	adapter := &Adapter{projection: func(context.Context, controlplane.Reservation, runtimesettings.RuntimeSettings, bool) (*engagementcontract.AllocationProjection, error) {
		return fixture.Engagement, nil
	}, materials: func(context.Context, *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error) {
		t.Fatal("rejected network policy decrypted credentials")
		return nil, nil
	}}
	_, err = adapter.WithPreparedSpec(t.Context(), controlplane.Reservation{}, fixture, func(control.AllocationSpec) (contracts.WorkerHandle, error) {
		t.Fatal("uncontrolled network allocation dispatched")
		return contracts.WorkerHandle{}, nil
	})
	if !errors.Is(err, controlplane.ErrInvalidRequest) {
		t.Fatalf("network policy error = %v", err)
	}
}

func TestSharedPreparationDerivesHTTPAuthorityFromActualWorkerTools(t *testing.T) {
	for _, tool := range []string{"read_artifact", "http_request", "http_credentials", "http_session_get", "scan_nuclei"} {
		t.Run(tool, func(t *testing.T) {
			spec, err := contracts.DecodePrivateStrict[control.AllocationSpec](contractstest.ReadFixture(t, "valid", "engagement-allocation-preflight.json"))
			if err != nil {
				t.Fatal(err)
			}
			toolset := "http-tools"
			if tool == "read_artifact" {
				toolset = "run-artifacts"
			}
			if tool == "scan_nuclei" {
				toolset = "scan"
			}
			spec.AgentTemplate.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: toolset, Version: "1"}, Tools: []string{tool}}}
			expected := tool != "read_artifact"
			projection := spec.Engagement
			if !expected {
				projection.HTTP = nil
			}
			materialCalls := 0
			adapter := &Adapter{
				projection: func(_ context.Context, _ controlplane.Reservation, _ runtimesettings.RuntimeSettings, usesHTTP bool) (*engagementcontract.AllocationProjection, error) {
					if usesHTTP != expected {
						t.Fatalf("HTTP owner=%v want=%v", usesHTTP, expected)
					}
					return projection, nil
				},
				materials: func(context.Context, *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error) {
					materialCalls++
					return nil, nil
				},
			}
			dispatched := false
			_, err = adapter.WithPreparedSpec(t.Context(), controlplane.Reservation{}, spec, func(prepared control.AllocationSpec) (contracts.WorkerHandle, error) {
				dispatched = true
				if (prepared.Engagement.HTTP != nil) != expected {
					t.Fatal("incorrect HTTP assignment")
				}
				return contracts.WorkerHandle{}, nil
			})
			if err != nil || !dispatched || (materialCalls == 1) != expected {
				t.Fatalf("dispatch=%v material=%d err=%v", dispatched, materialCalls, err)
			}
		})
	}
}

type prepareRPCFixture struct {
	pb.UnimplementedRuntimeLifecycleServer
	prepare func(context.Context, *pb.PrepareAllocationRequest) (*pb.PrepareAllocationResponse, error)
}

func (f *prepareRPCFixture) Prepare(ctx context.Context, request *pb.PrepareAllocationRequest) (*pb.PrepareAllocationResponse, error) {
	return f.prepare(ctx, request)
}
