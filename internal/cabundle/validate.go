package cabundle

import (
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"strings"
	"unicode/utf8"
)

var ErrInvalid = errors.New("invalid CA bundle")

var (
	beginCertificate = []byte("-----BEGIN CERTIFICATE-----")
	endCertificate   = []byte("-----END CERTIFICATE-----")
)

// Validate accepts one to eight X.509 certificate PEM blocks with only
// whitespace around or between them. Both Server publication and dispatch use
// this grammar so a published bundle cannot fail only at Runtime preparation.
func Validate(value string) error {
	if value == "" || len(value) > 64*1024 || !utf8.ValidString(value) ||
		strings.Contains(value, "PRIVATE KEY") {
		return ErrInvalid
	}
	rest := []byte(value)
	count := 0
	for len(bytes.TrimSpace(rest)) > 0 {
		rest = bytes.TrimSpace(rest)
		if !bytes.HasPrefix(rest, beginCertificate) {
			return ErrInvalid
		}
		end := bytes.Index(rest, endCertificate)
		if end <= len(beginCertificate) || !validBody(rest[len(beginCertificate):end]) {
			return ErrInvalid
		}
		block, remaining := pem.Decode(rest)
		if block == nil || block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			return ErrInvalid
		}
		if _, err := x509.ParseCertificate(block.Bytes); err != nil {
			return ErrInvalid
		}
		count++
		if count > 8 {
			return ErrInvalid
		}
		rest = remaining
	}
	if count == 0 {
		return ErrInvalid
	}
	return nil
}

func validBody(body []byte) bool {
	for _, char := range body {
		if (char >= 'A' && char <= 'Z') || (char >= 'a' && char <= 'z') ||
			(char >= '0' && char <= '9') || char == '+' || char == '/' ||
			char == '=' || char == '\r' || char == '\n' {
			continue
		}
		return false
	}
	return true
}
