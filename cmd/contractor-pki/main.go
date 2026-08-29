package main

import (
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"strings"

	"github.com/grauwolf32/contractor/internal/localpki"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "contractor-pki:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("expected init-ca, issue-control-plane, or issue-agent")
	}
	switch args[0] {
	case "init-ca":
		flags := newFlagSet("init-ca", stderr)
		root := flags.String("root", localpki.DefaultRoot, "local PKI directory")
		force := flags.Bool("force", false, "replace existing CA certificate and key")
		if err := parseFlags(flags, args[1:]); err != nil {
			return err
		}
		paths, err := (localpki.Generator{}).InitCA(*root, *force)
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintf(stdout, "created CA certificate %s and private key %s\n", paths.Certificate, paths.PrivateKey)
		return nil
	case "issue-control-plane":
		flags := newFlagSet("issue-control-plane", stderr)
		root := flags.String("root", localpki.DefaultRoot, "local PKI directory")
		uri := flags.String("uri", localpki.DefaultControlPlaneURI, "Control Plane URI SAN")
		dns := flags.String("dns", "localhost", "comma-separated DNS SANs")
		ips := flags.String("ip", "127.0.0.1,::1", "comma-separated IP SANs")
		force := flags.Bool("force", false, "replace existing leaf certificate and key")
		if err := parseFlags(flags, args[1:]); err != nil {
			return err
		}
		leaf, err := leafOptions(*dns, *ips, *force)
		if err != nil {
			return err
		}
		paths, err := (localpki.Generator{}).IssueControlPlane(*root, localpki.ControlPlaneOptions{LeafOptions: leaf, URI: *uri})
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintf(stdout, "created Control Plane certificate %s and private key %s\n", paths.Certificate, paths.PrivateKey)
		return nil
	case "issue-agent":
		flags := newFlagSet("issue-agent", stderr)
		root := flags.String("root", localpki.DefaultRoot, "local PKI directory")
		name := flags.String("name", "agent-local", "Runtime Agent certificate name")
		dns := flags.String("dns", "localhost", "comma-separated DNS SANs")
		ips := flags.String("ip", "127.0.0.1,::1", "comma-separated IP SANs")
		force := flags.Bool("force", false, "replace existing leaf certificate and key")
		if err := parseFlags(flags, args[1:]); err != nil {
			return err
		}
		leaf, err := leafOptions(*dns, *ips, *force)
		if err != nil {
			return err
		}
		paths, err := (localpki.Generator{}).IssueAgent(*root, *name, leaf)
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintf(stdout, "created Runtime Agent certificate %s and private key %s\n", paths.Certificate, paths.PrivateKey)
		return nil
	default:
		return fmt.Errorf("unknown operation %q", args[0])
	}
}

func newFlagSet(name string, stderr io.Writer) *flag.FlagSet {
	result := flag.NewFlagSet("contractor-pki "+name, flag.ContinueOnError)
	result.SetOutput(stderr)
	return result
}

func parseFlags(flags *flag.FlagSet, args []string) error {
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	return nil
}

func leafOptions(rawDNS, rawIPs string, force bool) (localpki.LeafOptions, error) {
	dnsNames := splitNonEmpty(rawDNS)
	ipAddresses := make([]net.IP, 0)
	for _, raw := range splitNonEmpty(rawIPs) {
		address := net.ParseIP(raw)
		if address == nil {
			return localpki.LeafOptions{}, fmt.Errorf("invalid IP SAN %q", raw)
		}
		ipAddresses = append(ipAddresses, address)
	}
	return localpki.LeafOptions{DNSNames: dnsNames, IPAddresses: ipAddresses, Force: force}, nil
}

func splitNonEmpty(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		if value := strings.TrimSpace(part); value != "" {
			result = append(result, value)
		}
	}
	return result
}
