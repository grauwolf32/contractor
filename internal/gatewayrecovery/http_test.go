package gatewayrecovery

import (
	"bytes"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestSendMarksOnlyDeliveredTimeoutsAbandoned(t *testing.T) {
	release := make(chan struct{})
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	defer slow.Close()
	defer close(release)
	closed, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	unreachable := "http://" + closed.Addr().String()
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name string
		url  string
		want Failure
	}{
		// The Gateway holds the request: it may still be generating the answer.
		{name: "delivered", url: slow.URL, want: Failure{Code: "gateway_timeout", Retryable: true, Abandoned: true}},
		{name: "unreachable", url: unreachable, want: Failure{Code: "gateway_unavailable", Retryable: true}},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, err := http.NewRequest(http.MethodPost, test.url, bytes.NewReader([]byte(`{}`)))
			if err != nil {
				t.Fatal(err)
			}
			client := &http.Client{Timeout: 200 * time.Millisecond}
			_, err = Send(request, client, 1024, contracts.DefaultGatewayFailureSignatures())
			var failure *FailureError
			if !errors.As(err, &failure) || failure.Failure != test.want {
				t.Fatalf("Send error = %v, want %+v", err, test.want)
			}
		})
	}
}
