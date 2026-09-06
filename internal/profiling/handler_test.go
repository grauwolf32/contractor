package profiling

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHandlerExposesOnlyTheExplicitProfileAllowlist(t *testing.T) {
	handler := NewHandler()
	response := request(handler, http.MethodGet, "/debug/pprof/", nil)
	if response.Code != http.StatusOK {
		t.Fatalf("index status = %d", response.Code)
	}
	index := response.Body.String()
	for _, name := range []string{"profile", "heap", "allocs", "goroutine", "threadcreate", "trace", "symbol"} {
		if !strings.Contains(index, ">"+name+"<") {
			t.Fatalf("index does not include %q: %s", name, index)
		}
	}
	for _, name := range []string{"cmdline", "block", "mutex"} {
		if strings.Contains(index, name) {
			t.Fatalf("index exposes forbidden profile %q: %s", name, index)
		}
	}
	for _, path := range []string{
		"/debug/pprof/cmdline",
		"/debug/pprof/block",
		"/debug/pprof/mutex",
		"/debug/pprof/heap/",
		"/debug/pprof/unknown",
	} {
		if got := request(handler, http.MethodGet, path, nil).Code; got != http.StatusNotFound {
			t.Fatalf("GET %s status = %d, want 404", path, got)
		}
	}
	if got := request(handler, http.MethodPost, "/debug/pprof/heap", nil); got.Code != http.StatusMethodNotAllowed || got.Header().Get("Allow") != "GET" {
		t.Fatalf("POST heap = (%d, %q)", got.Code, got.Header().Get("Allow"))
	}
	if got := request(handler, http.MethodGet, "/debug/pprof/?debug=1", nil).Code; got != http.StatusBadRequest {
		t.Fatalf("index query status = %d, want 400", got)
	}
	if got := request(handler, http.MethodGet, "/debug/pprof/symbol?0x0", nil).Code; got != http.StatusOK {
		t.Fatalf("symbol status = %d", got)
	}
}

func TestHandlerValidatesEveryDurationAndSnapshotQuery(t *testing.T) {
	handler := NewHandler()
	for _, target := range []string{
		"/debug/pprof/profile?seconds=0",
		"/debug/pprof/profile?seconds=61",
		"/debug/pprof/profile?seconds=-1",
		"/debug/pprof/profile?seconds=1.5",
		"/debug/pprof/profile?seconds=1&seconds=2",
		"/debug/pprof/profile?unknown=1",
		"/debug/pprof/trace?seconds=0",
		"/debug/pprof/trace?seconds=11",
		"/debug/pprof/heap?seconds=61",
		"/debug/pprof/heap?seconds=1&gc=1",
		"/debug/pprof/heap?seconds=1&debug=1",
		"/debug/pprof/goroutine?debug=3",
		"/debug/pprof/allocs?gc=1",
	} {
		if got := request(handler, http.MethodGet, target, nil).Code; got != http.StatusBadRequest {
			t.Fatalf("GET %s status = %d, want 400", target, got)
		}
	}
	for _, target := range []string{
		"/debug/pprof/heap",
		"/debug/pprof/heap?gc=1",
		"/debug/pprof/allocs?debug=1",
		"/debug/pprof/goroutine?debug=2",
		"/debug/pprof/threadcreate",
	} {
		if got := request(handler, http.MethodGet, target, nil).Code; got != http.StatusOK {
			t.Fatalf("GET %s status = %d, want 200", target, got)
		}
	}
}

func TestHandlerBoundsSymbolLookupInput(t *testing.T) {
	handler := NewHandler()
	tooLarge := bytes.Repeat([]byte("1"), maxSymbolRequestBytes+1)
	response := request(handler, http.MethodPost, "/debug/pprof/symbol", bytes.NewReader(tooLarge))
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized symbol status = %d", response.Code)
	}
	response = request(handler, http.MethodPost, "/debug/pprof/symbol", strings.NewReader("0x0"))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "num_symbols") {
		t.Fatalf("bounded symbol response = (%d, %q)", response.Code, response.Body.String())
	}
}

func TestTimedAndSnapshotCapacityDoesNotQueue(t *testing.T) {
	handler := NewHandler()
	profileContext, cancelProfile := context.WithCancel(context.Background())
	profileDone := make(chan int, 1)
	go func() {
		request := httptest.NewRequest(http.MethodGet, "/debug/pprof/profile?seconds=60", nil).WithContext(profileContext)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		profileDone <- response.Code
	}()
	waitForSlot(t, handler.timed)
	if got := request(handler, http.MethodGet, "/debug/pprof/trace?seconds=1", nil).Code; got != http.StatusConflict {
		t.Fatalf("concurrent timed capture status = %d", got)
	}
	if got := request(handler, http.MethodGet, "/debug/pprof/heap", nil).Code; got != http.StatusOK {
		t.Fatalf("independent snapshot status = %d", got)
	}
	cancelProfile()
	waitForCompletion(t, profileDone)
	waitForReleasedSlot(t, handler.timed)

	deltaContext, cancelDelta := context.WithCancel(context.Background())
	deltaDone := make(chan int, 1)
	go func() {
		request := httptest.NewRequest(http.MethodGet, "/debug/pprof/heap?seconds=60", nil).WithContext(deltaContext)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		deltaDone <- response.Code
	}()
	waitForSlot(t, handler.snapshot)
	if got := request(handler, http.MethodGet, "/debug/pprof/allocs", nil).Code; got != http.StatusConflict {
		t.Fatalf("concurrent snapshot status = %d", got)
	}
	cancelDelta()
	waitForCompletion(t, deltaDone)
	waitForReleasedSlot(t, handler.snapshot)
	if got := request(handler, http.MethodGet, "/debug/pprof/allocs", nil).Code; got != http.StatusOK {
		t.Fatalf("reused snapshot slot status = %d", got)
	}
}

func TestCPUAndTraceOutputIsReadableByGoTools(t *testing.T) {
	handler := NewHandler()
	checks := []struct {
		target string
		tool   []string
	}{
		{target: "/debug/pprof/profile?seconds=1", tool: []string{"pprof", "-top"}},
		{target: "/debug/pprof/trace?seconds=1", tool: []string{"trace", "-d=parsed"}},
	}
	for _, check := range checks {
		response := request(handler, http.MethodGet, check.target, nil)
		if response.Code != http.StatusOK || response.Body.Len() == 0 {
			t.Fatalf("GET %s = (%d, %d bytes)", check.target, response.Code, response.Body.Len())
		}
		path := filepath.Join(t.TempDir(), "profile.bin")
		if err := os.WriteFile(path, response.Body.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		arguments := append([]string{"tool"}, check.tool...)
		arguments = append(arguments, path)
		command := exec.Command("go", arguments...)
		command.Stdout = io.Discard
		var stderr bytes.Buffer
		command.Stderr = &stderr
		if err := command.Run(); err != nil {
			t.Fatalf("go tool for %s: %v: %s", check.target, err, stderr.String())
		}
	}
}

func request(handler http.Handler, method string, target string, body io.Reader) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, target, body)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func waitForSlot(t *testing.T, slot requestSlot) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !slot.active() {
		if time.Now().After(deadline) {
			t.Fatal("profiling slot was not acquired")
		}
		time.Sleep(time.Millisecond)
	}
}

func waitForReleasedSlot(t *testing.T, slot requestSlot) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for slot.active() {
		if time.Now().After(deadline) {
			t.Fatal("profiling slot was not released")
		}
		time.Sleep(time.Millisecond)
	}
}

func waitForCompletion(t *testing.T, result <-chan int) {
	t.Helper()
	select {
	case <-result:
	case <-time.After(2 * time.Second):
		t.Fatal("profiling request did not stop after cancellation")
	}
}
