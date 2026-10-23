// A real upstream Interactsh server on disposable loopback ports. It owns no
// Contractor state; the parent process terminates it after the protocol test.
package main

import (
	"encoding/json"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/projectdiscovery/interactsh/pkg/server"
	"github.com/projectdiscovery/interactsh/pkg/storage"
)

func main() {
	if len(os.Args) != 4 {
		panic("expected HTTP, DNS and SMTP ports")
	}
	ports := make([]int, 3)
	for i, arg := range os.Args[1:] {
		value, err := strconv.Atoi(arg)
		if err != nil {
			panic(err)
		}
		ports[i] = value
	}
	store, err := storage.New(&storage.Options{MaxSize: 32, EvictionTTL: time.Hour})
	if err != nil {
		panic(err)
	}
	defer store.Close()
	options := &server.Options{Domains: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}, ListenIP: "127.0.0.1", HttpPort: ports[0], DnsPort: ports[1], SmtpPort: ports[2], Storage: store, Stats: &server.Metrics{}, CorrelationIdLength: 20, CorrelationIdNonceLength: 13, Token: "local-protocol-test-token", OriginURL: "*"}
	httpServer, err := server.NewHTTPServer(options)
	if err != nil {
		panic(err)
	}
	smtpServer, err := server.NewSMTPServer(options)
	if err != nil {
		panic(err)
	}
	go httpServer.ListenAndServe(nil, make(chan bool, 8), make(chan bool, 8))
	go server.NewDNSServer("udp", options).ListenAndServe(make(chan bool, 8))
	go smtpServer.ListenAndServe(nil, make(chan bool, 8), make(chan bool, 8))
	if err := json.NewEncoder(os.Stdout).Encode(map[string]int{"http": ports[0], "dns": ports[1], "smtp": ports[2]}); err != nil {
		panic(err)
	}
	select {}
}
