package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
)

func TestPublicEventWebSocketRequiresCookieAndExactOrigin(t *testing.T) {
	fixture := newHandlerFixture(t)
	server := httptest.NewServer(fixture.handler)
	t.Cleanup(server.Close)
	websocketURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/v1/events/ws"
	dialer := websocket.Dialer{
		Subprotocols: []string{publicevents.ProtocolVersion}, HandshakeTimeout: 2 * time.Second,
	}
	for name, header := range map[string]http.Header{
		"bearer is not a browser session": {
			"Origin":        []string{testBrowserOrigin},
			"Authorization": []string{"Bearer " + testBearerToken},
		},
		"missing Origin":    {},
		"unexpected Origin": {"Origin": []string{"https://attacker.test"}},
	} {
		t.Run(name, func(t *testing.T) {
			connection, response, err := dialer.Dial(websocketURL, header)
			if connection != nil {
				connection.Close()
			}
			if err == nil || response == nil {
				t.Fatalf("unauthorized WebSocket = connection=%v response=%v error=%v", connection, response, err)
			}
			defer response.Body.Close()
			wantStatus := http.StatusForbidden
			if name == "bearer is not a browser session" || name == "missing Origin" {
				wantStatus = http.StatusUnauthorized
			}
			if response.StatusCode != wantStatus {
				t.Fatalf("unauthorized WebSocket status = %d", response.StatusCode)
			}
		})
	}

	loginRequest, err := http.NewRequest(
		http.MethodPost,
		server.URL+"/v1/auth/login",
		bytes.NewBufferString(`{"username":"admin","password":"correct horse battery staple"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	loginRequest.Header.Set("Content-Type", "application/json")
	loginRequest.Header.Set("Origin", testBrowserOrigin)
	loginResponse, err := http.DefaultClient.Do(loginRequest)
	if err != nil {
		t.Fatal(err)
	}
	defer loginResponse.Body.Close()
	if loginResponse.StatusCode != http.StatusOK || len(loginResponse.Cookies()) != 1 {
		t.Fatalf("browser login = %d cookies=%v", loginResponse.StatusCode, loginResponse.Cookies())
	}
	header := http.Header{"Origin": []string{testBrowserOrigin}}
	header.Set("Cookie", loginResponse.Cookies()[0].Name+"="+loginResponse.Cookies()[0].Value)
	connection, response, err := dialer.Dial(websocketURL, header)
	if err != nil {
		status := 0
		if response != nil {
			status = response.StatusCode
			response.Body.Close()
		}
		t.Fatalf("authenticated WebSocket status=%d error=%v", status, err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	if connection.Subprotocol() != publicevents.ProtocolVersion {
		t.Fatalf("negotiated subprotocol = %q", connection.Subprotocol())
	}
	if err := connection.WriteJSON(map[string]any{
		"version": publicevents.ProtocolVersion, "type": "subscribe",
		"subscriptionId": "operations", "stream": map[string]any{"kind": "operations"},
	}); err != nil {
		t.Fatal(err)
	}
	_ = connection.SetReadDeadline(time.Now().Add(2 * time.Second))
	var frame map[string]any
	if err := connection.ReadJSON(&frame); err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(frame)
	if frame["type"] != "subscribed" || strings.Contains(string(encoded), testBearerToken) ||
		strings.Contains(string(encoded), loginResponse.Cookies()[0].Value) {
		t.Fatalf("authenticated subscription frame = %s", encoded)
	}
}
