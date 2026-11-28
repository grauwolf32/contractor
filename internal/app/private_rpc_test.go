package app

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/settingsstore"

	"github.com/grauwolf32/contractor/internal/runtimewire"
)

type unreadRPCBody struct{}

func (unreadRPCBody) Read([]byte) (int, error) { panic("saturated RPC read a body") }
func (unreadRPCBody) Close() error             { return nil }

func TestPrivateRPCFamiliesShareProcessAdmissionBeforeDecode(t *testing.T) {
	entered := make(chan struct{}, runtimewire.MaxConcurrentRPCs)
	returned := make(chan *httptest.ResponseRecorder, 2*runtimewire.MaxConcurrentRPCs)
	release := make(chan struct{})
	handler := newPrivateHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { entered <- struct{}{}; <-release }))
	var calls sync.WaitGroup
	defer func() { close(release); calls.Wait() }()
	dispatch := func(path string, body bool) {
		calls.Add(1)
		go func() {
			defer calls.Done()
			request := httptest.NewRequest(http.MethodPost, path, nil)
			if body {
				request.Body = unreadRPCBody{}
			}
			request.ProtoMajor = 2
			request.Header.Set("Content-Type", "application/grpc")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			returned <- response
		}()
	}
	admitted := func() {
		select {
		case <-entered:
		case response := <-returned:
			t.Fatalf("RPC rejected before its bound: %v", response.Header())
		case <-t.Context().Done():
			t.Fatal("admission setup cancelled")
		}
	}
	rejected := func(path string) {
		dispatch(path, true)
		select {
		case response := <-returned:
			if response.Header().Get("Grpc-Status") != "8" || response.Header().Get("Grpc-Status-Details-Bin") == "" {
				t.Fatalf("capacity=%v", response.Header())
			}
		case <-entered:
			t.Fatal("exhausted RPC reached the decoder")
		case <-t.Context().Done():
			t.Fatal("rejection did not finish")
		}
	}
	const domain = "/contractor.runtime.v1alpha1.Allocation/PlannerOperation"
	const control = "/contractor.runtime.v1alpha1.ControlPlane/Heartbeat"
	for range runtimewire.MaxConcurrentRPCs - privateControlRPCReserve - privateWorkerRPCReserve {
		dispatch(domain, false)
		admitted()
	}
	rejected(domain)
	for range privateWorkerRPCReserve {
		dispatch(pb.Artifact_Read_FullMethodName, false)
		admitted()
	}
	rejected(pb.Artifact_Read_FullMethodName)
	// Waiting Planner callbacks leave both Worker and control positions free.
	for range privateControlRPCReserve {
		dispatch(control, false)
		admitted()
	}
	rejected(domain)
	rejected(control)
}

func TestPrivateHTTPPayloadRoutesAreRemoved(t *testing.T) {
	handler := newPrivateHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("private HTTP reached an RPC handler") }))
	for _, test := range []struct{ method, path string }{
		{http.MethodGet, "/private/v1/allocations/a/artifacts/b/c"},
		{http.MethodPut, "/private/v1/allocations/a/artifacts/b/c"},
		{http.MethodPost, "/private/v1/allocations/a/finding-proposals"},
	} {
		request := httptest.NewRequest(test.method, test.path, nil)
		request.Body = unreadRPCBody{}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound {
			t.Fatalf("old private route %s = %d", test.path, response.Code)
		}
	}
}

func TestWaitingPlannersLeaveDependentWorkerRPCsAvailable(t *testing.T) {
	entered := make(chan struct{}, settingsstore.MaximumConcurrentRuns)
	release := make(chan struct{})
	var calls sync.WaitGroup
	handler := newPrivateHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == pb.Allocation_PlannerOperation_FullMethodName {
			entered <- struct{}{}
			<-release
		}
		w.Header().Set("Grpc-Status", "0")
	}))
	request := func(path string) *http.Request {
		r := httptest.NewRequest(http.MethodPost, path, nil)
		r.ProtoMajor = 2
		r.Header.Set("Content-Type", "application/grpc")
		return r
	}
	defer func() { close(release); calls.Wait() }()
	// Try every supported model lane, keeping every admitted Planner waiting
	// for its Worker. Surplus must fail before consuming progress capacity.
	for range settingsstore.MaximumConcurrentRuns {
		returned := make(chan *httptest.ResponseRecorder, 1)
		calls.Go(func() {
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, request(pb.Allocation_PlannerOperation_FullMethodName))
			returned <- w
		})
		select {
		case <-entered:
		case w := <-returned:
			if w.Header().Get("Grpc-Status") != "8" || w.Header().Get("Grpc-Status-Details-Bin") == "" {
				t.Fatalf("invalid rejection: %v", w.Header())
			}
		case <-t.Context().Done():
			t.Fatal("Planner admission cancelled")
		}
	}
	for _, path := range []string{
		pb.Artifact_Read_FullMethodName, pb.Artifact_Write_FullMethodName,
		pb.Allocation_ListArtifacts_FullMethodName, pb.Allocation_GatewayRecovery_FullMethodName,
		pb.ControlPlane_Heartbeat_FullMethodName,
	} {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, request(path))
		if w.Header().Get("Grpc-Status") != "0" {
			t.Fatalf("waiting Planners block dependent RPC %s: %v", path, w.Header())
		}
	}
}
