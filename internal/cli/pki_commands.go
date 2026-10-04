package cli

import (
	"flag"
	"fmt"

	"github.com/grauwolf32/contractor/internal/localpki"
)

type pkiResult struct {
	Certificate   string `json:"certificate"`
	PrivateKey    string `json:"privateKey"`
	CACertificate string `json:"caCertificate,omitempty"`
}

func (c *CLI) runPKI(args []string, printer *Printer) error {
	if len(args) == 0 {
		return &UsageError{Message: "pki requires init-ca, issue-control-plane, issue-runtime, renew-control-plane, or renew-runtime"}
	}
	var paths localpki.Paths
	var caCertificate string
	switch args[0] {
	case "init-ca":
		var root string
		var force bool
		flags, err := parseFlags("contractor pki init-ca", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.StringVar(&root, "root", localpki.DefaultRoot, "local PKI directory")
			flags.BoolVar(&force, "force", false, "replace the existing CA certificate and key")
		})
		if err != nil || flags == nil {
			return err
		}
		if _, err := requirePositionals(flags, 0, 0, "usage: contractor pki init-ca [--root DIR] [--force]"); err != nil {
			return err
		}
		paths, err = (localpki.Generator{}).InitCA(root, force)
		if err != nil {
			return err
		}
	case "issue-control-plane":
		var root, uri, rawDNS, rawIPs string
		var force bool
		flags, err := parseFlags("contractor pki issue-control-plane", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.StringVar(&root, "root", localpki.DefaultRoot, "local PKI directory")
			flags.StringVar(&uri, "uri", localpki.DefaultControlPlaneURI, "Control Plane URI SAN")
			flags.StringVar(&rawDNS, "dns", "localhost", "comma-separated DNS SANs")
			flags.StringVar(&rawIPs, "ip", "127.0.0.1,::1", "comma-separated IP SANs")
			flags.BoolVar(&force, "force", false, "replace the existing certificate and key")
		})
		if err != nil || flags == nil {
			return err
		}
		if _, err := requirePositionals(flags, 0, 0, "usage: contractor pki issue-control-plane [flags]"); err != nil {
			return err
		}
		leaf, err := parseLeafOptions(rawDNS, rawIPs, force)
		if err != nil {
			return err
		}
		paths, err = (localpki.Generator{}).IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf, URI: uri})
		if err != nil {
			return err
		}
		caCertificate = localpki.CAPaths(root).Certificate
	case "issue-runtime", "issue-agent":
		var root, name, rawDNS, rawIPs string
		var force bool
		flags, err := parseFlags("contractor pki issue-runtime", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.StringVar(&root, "root", localpki.DefaultRoot, "local PKI directory")
			flags.StringVar(&name, "name", "runtime-local", "Runtime certificate name")
			flags.StringVar(&rawDNS, "dns", "localhost", "comma-separated DNS SANs")
			flags.StringVar(&rawIPs, "ip", "127.0.0.1,::1", "comma-separated IP SANs")
			flags.BoolVar(&force, "force", false, "replace the existing certificate and key")
		})
		if err != nil || flags == nil {
			return err
		}
		if _, err := requirePositionals(flags, 0, 0, "usage: contractor pki issue-runtime [--name NAME] [flags]"); err != nil {
			return err
		}
		leaf, err := parseLeafOptions(rawDNS, rawIPs, force)
		if err != nil {
			return err
		}
		paths, err = (localpki.Generator{}).IssueAgent(root, name, leaf)
		if err != nil {
			return err
		}
		caCertificate = localpki.CAPaths(root).Certificate
	case "renew-control-plane":
		var root string
		flags, err := parseFlags("contractor pki renew-control-plane", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.StringVar(&root, "root", localpki.DefaultRoot, "local PKI directory")
		})
		if err != nil || flags == nil {
			return err
		}
		if _, err := requirePositionals(flags, 0, 0, "usage: contractor pki renew-control-plane [--root DIR]"); err != nil {
			return err
		}
		paths, err = (localpki.Generator{}).RenewControlPlane(root)
		if err != nil {
			return err
		}
		caCertificate = localpki.CAPaths(root).Certificate
	case "renew-runtime", "renew-agent":
		var root, name string
		flags, err := parseFlags("contractor pki renew-runtime", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.StringVar(&root, "root", localpki.DefaultRoot, "local PKI directory")
			flags.StringVar(&name, "name", "runtime-local", "Runtime certificate name")
		})
		if err != nil || flags == nil {
			return err
		}
		if _, err := requirePositionals(flags, 0, 0, "usage: contractor pki renew-runtime [--name NAME] [--root DIR]"); err != nil {
			return err
		}
		paths, err = (localpki.Generator{}).RenewAgent(root, name)
		if err != nil {
			return err
		}
		caCertificate = localpki.CAPaths(root).Certificate
	default:
		return &UsageError{Message: fmt.Sprintf("unknown pki command %q", args[0])}
	}

	result := pkiResult{Certificate: paths.Certificate, PrivateKey: paths.PrivateKey, CACertificate: caCertificate}
	if printer.Mode() == OutputTable {
		if result.CACertificate != "" {
			return printer.Table([]string{"CERTIFICATE", "PRIVATE KEY", "CA CERTIFICATE"}, [][]string{{result.Certificate, result.PrivateKey, result.CACertificate}})
		}
		return printer.Table([]string{"CERTIFICATE", "PRIVATE KEY"}, [][]string{{result.Certificate, result.PrivateKey}})
	}
	return printer.Object(result, result.Certificate)
}

func parseLeafOptions(rawDNS, rawIPs string, force bool) (localpki.LeafOptions, error) {
	leaf, err := localpki.ParseLeafOptions(rawDNS, rawIPs, force)
	if err != nil {
		return localpki.LeafOptions{}, &UsageError{Message: err.Error()}
	}
	return leaf, nil
}
