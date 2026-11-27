package app

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/runtimewire"
)

type unreadRPCBody struct{}

func (unreadRPCBody) Read([]byte) (int, error) { panic("saturated RPC read a body") }
func (unreadRPCBody) Close() error             { return nil }

func TestPrivateRPCFamiliesShareProcessAdmissionBeforeDecode(t *testing.T) {
	entered := make(chan struct{}, runtimewire.MaxConcurrentRPCs)
	returned := make(chan *httptest.ResponseRecorder, 2*runtimewire.MaxConcurrentRPCs)
	release := make(chan struct{})
	handler := newPrivateHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { entered <- struct{}{}; <-release }), http.NotFoundHandler())
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
	for range runtimewire.MaxConcurrentRPCs - 4 {
		dispatch(domain, false)
		admitted()
	}
	rejected(domain)
	// Long domain calls leave the four specified control positions available.
	for range 4 {
		dispatch(control, false)
		admitted()
	}
	rejected(domain)
	rejected(control)
}
