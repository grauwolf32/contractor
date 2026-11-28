package privateartifacts

import (
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPrivatePayloadNativeMethodsNeverReachStorage(t *testing.T) {
	repository := newMemoryRepository()
	native, err := NewGRPCHandler(Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository), Transfers: artifactpgtest.NewRuntime()})
	if err != nil {
		t.Fatal(err)
	}
	for _, method := range []string{http.MethodGet, http.MethodHead} {
		for _, path := range []string{pb.Artifact_Read_FullMethodName, pb.Artifact_Write_FullMethodName, pb.Artifact_SubmitFinding_FullMethodName} {
			request := trustedRequest(method, path, unreadAllocationBody{})
			request.ProtoMajor = 2
			request.ProtoMinor = 0
			request.Header.Set("Content-Type", "application/grpc")
			request.Header.Set(allocationIDHeader, "allocation-1")
			response := httptest.NewRecorder()
			native.ServeHTTP(response, request)
			if response.Code != http.StatusMethodNotAllowed {
				t.Fatalf("%s %s = %d", method, path, response.Code)
			}
		}
	}
	if repository.readCalls+repository.writeCalls+repository.metadataCalls != 0 {
		t.Fatal("invalid RPC method reached storage")
	}
}
