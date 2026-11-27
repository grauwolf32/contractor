package controlplane

// Existing domain rejection matrices use raw JSON documents (including malformed
// telemetry). This test fixture serves them inside native protobuf responses over
// real TLS/gRPC. The callback is a document fixture builder, never a production
// HTTP endpoint or a fallback transport.
import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
)

var runtimeFixturePrincipals sync.Map
var runtimeFixtureCleanup sync.Map

func fixturePrincipal(origin string) string {
	if value, ok := runtimeFixturePrincipals.Load(origin); ok {
		return value.(string)
	}
	return strings.Repeat("a", 64)
}
func newRuntimeFixture(t *testing.T, documents http.Handler) *httptest.Server {
	t.Helper()
	rpc := grpc.NewServer(grpc.MaxSendMsgSize(2 << 20))
	fixture := &runtimeFixture{documents: documents}
	pb.RegisterRuntimeLifecycleServer(rpc, fixture)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			rpc.ServeHTTP(w, r)
			return
		}
		if r.Method == http.MethodGet {
			documents.ServeHTTP(w, r)
			return
		}
		http.NotFound(w, r)
	}))
	server.EnableHTTP2 = true
	server.StartTLS()
	fixture.origin = server.URL
	principal, err := mtls.RuntimeAgentID(server.Certificate())
	if err != nil {
		t.Fatal(err)
	}
	runtimeFixturePrincipals.Store(server.URL, principal)
	runtimeFixtureCleanup.Store(server.Client(), t.Cleanup)
	t.Cleanup(func() {
		runtimeFixturePrincipals.Delete(server.URL)
		runtimeFixtureCleanup.Delete(server.Client())
		server.Close()
		rpc.Stop()
	})
	return server
}

type runtimeFixture struct {
	pb.UnimplementedRuntimeLifecycleServer
	documents http.Handler
	origin    string
}

func (f *runtimeFixture) document(ctx context.Context, op, id string, value contracts.Validatable) ([]byte, error) {
	var raw []byte
	var err error
	if revealed, ok := value.(interface{ RevealJSON() ([]byte, error) }); ok {
		raw, err = revealed.RevealJSON()
	} else {
		raw, err = json.Marshal(value)
	}
	if err != nil {
		return nil, err
	}
	request := httptest.NewRequest(http.MethodPost, f.origin+"/private/v1/allocations/"+id+"/"+op, bytes.NewReader(raw))
	request.TLS = &tls.ConnectionState{}
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		for _, value := range md.Get("x-request-id") {
			request.Header.Add("X-Request-ID", value)
		}
	}
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	f.documents.ServeHTTP(response, request)
	if response.Code >= 400 {
		return nil, runtimewire.Error(ctx, codes.InvalidArgument, "fixture_rejection", false)
	}
	return response.Body.Bytes(), nil
}
func (f *runtimeFixture) Prepare(ctx context.Context, request *pb.PrepareAllocationRequest) (*pb.PrepareAllocationResponse, error) {
	var value control.PrepareAllocationRequest
	if err := runtimewire.Decode(request, &value); err != nil {
		return nil, err
	}
	raw, err := f.document(ctx, "prepare", value.Spec.AllocationID, value)
	if err != nil {
		return nil, err
	}
	var response control.PrepareAllocationResponse
	result := &pb.PrepareAllocationResponse{}
	if contracts.DecodePrivate(raw, &response) != nil || runtimewire.Encode(response, result) != nil {
		result.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
	}
	return result, nil
}
func (f *runtimeFixture) Finalize(ctx context.Context, request *pb.FinalizeAllocationRequest) (*pb.AllocationFinalResponse, error) {
	var value control.FinalizeAllocationRequest
	if err := runtimewire.Decode(request, &value); err != nil {
		return nil, err
	}
	raw, err := f.document(ctx, "finalize", value.AllocationID, value)
	if err != nil {
		return nil, err
	}
	return finalDocument(raw), nil
}
func (f *runtimeFixture) Abort(ctx context.Context, request *pb.AbortAllocationRequest) (*pb.AllocationFinalResponse, error) {
	var value control.AbortAllocationRequest
	if err := runtimewire.Decode(request, &value); err != nil {
		return nil, err
	}
	raw, err := f.document(ctx, "abort", value.AllocationID, value)
	if err != nil {
		return nil, err
	}
	return finalDocument(raw), nil
}
func (f *runtimeFixture) Release(ctx context.Context, request *pb.ReleaseAllocationRequest) (*pb.ReleaseAllocationResponse, error) {
	var value control.ReleaseAllocationRequest
	if err := runtimewire.Decode(request, &value); err != nil {
		return nil, err
	}
	raw, err := f.document(ctx, "release", value.AllocationID, value)
	if err != nil {
		return nil, err
	}
	result := &pb.ReleaseAllocationResponse{}
	if len(raw) != 0 {
		result.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
	}
	return result, nil
}
func finalDocument(raw []byte) *pb.AllocationFinalResponse {
	// Decode only the envelope, retaining received report document bytes so the
	// production decoder, not this fixture, decides telemetry salvage and bounds.
	var value struct {
		APIVersion string `json:"apiVersion"`
		Report     struct {
			ReportID     string          `json:"reportId"`
			AllocationID string          `json:"allocationId"`
			StartedAt    string          `json:"startedAt"`
			FinishedAt   string          `json:"finishedAt"`
			Worker       json.RawMessage `json:"worker"`
			Runtime      json.RawMessage `json:"runtime"`
		} `json:"report"`
	}
	if strictjson.Decode(raw, &value) != nil {
		result := &pb.AllocationFinalResponse{}
		result.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
		return result
	}
	return &pb.AllocationFinalResponse{ApiVersion: proto.String(value.APIVersion), Report: &pb.AllocationFinalReport{
		ReportId: proto.String(value.Report.ReportID), AllocationId: proto.String(value.Report.AllocationID), StartedAt: proto.String(value.Report.StartedAt), FinishedAt: proto.String(value.Report.FinishedAt),
		Worker: &pb.ExecutionReportDocument{Json: value.Report.Worker}, Runtime: &pb.RuntimeReportDocument{Json: value.Report.Runtime},
	}}
}
func fixtureRuntimeClient(client *http.Client) (*RuntimeControlClient, error) {
	result := &RuntimeControlClient{client: client, timeout: 3 * time.Second}
	if tr, ok := client.Transport.(*http.Transport); ok && tr.TLSClientConfig != nil {
		channels, err := mtls.NewRuntimeChannels(tr.TLSClientConfig)
		if err != nil {
			return nil, err
		}
		result.channels = channels
		if register, ok := runtimeFixtureCleanup.Load(client); ok {
			register.(func(func()))(result.Close)
		}
	}
	return result, nil
}
