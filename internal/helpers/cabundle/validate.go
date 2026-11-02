package cabundle

import (
	"crypto/x509"
	"encoding/base64"
	"errors"
	"regexp"
	"strings"
	"unicode/utf8"
)

var ErrInvalid = errors.New("invalid CA bundle")

// certificateBlock and the bundle pattern below are byte-for-byte the same
// expressions as runtime/src/contractor_runtime/contracts/base.py. Every
// construct is ASCII-only, so Go RE2 and Python re agree on what they match.
const certificateBlock = `-----BEGIN CERTIFICATE-----\r?\n((?:[A-Za-z0-9+/=]+\r?\n)+)-----END CERTIFICATE-----`

var (
	bundlePattern = regexp.MustCompile(`^(?:[\t\n\f\r ]*\n)?` + certificateBlock +
		`(?:[\t\n\f\r ]*\n` + certificateBlock + `)*[\t\n\f\r ]*$`)
	certificateBlockPattern = regexp.MustCompile(certificateBlock)
	lineBreaks              = strings.NewReplacer("\r", "", "\n", "")
)

// Validate accepts one to eight X.509 CERTIFICATE PEM blocks using the same
// grammar as Runtime contract validation (_validate_ca_bundle), so Runtime
// allocation preparation never rejects a published bundle for its PEM
// structure, and the grammar stays within what OpenSSL loads as one bundle:
//
//   - each -----BEGIN CERTIFICATE----- and -----END CERTIFICATE----- marker
//     starts a line, and the BEGIN line ends right after its marker with LF
//     or CRLF;
//   - between the markers are one or more non-empty base64 lines
//     ([A-Za-z0-9+/=]) ending in LF or CRLF, with no blank lines and no PEM
//     headers;
//   - only ASCII whitespace ([\t\n\f\r ]) appears before, between and after
//     blocks, and every BEGIN marker starts a new line;
//   - each block's base64 decodes (padding required) to a DER certificate that
//     x509.ParseCertificate accepts.
//
// The bundle is also limited to 64 KiB of valid UTF-8 without any PRIVATE KEY
// text.
func Validate(value string) error {
	if value == "" || len(value) > 64*1024 || !utf8.ValidString(value) ||
		strings.Contains(value, "PRIVATE KEY") || !bundlePattern.MatchString(value) {
		return ErrInvalid
	}
	blocks := certificateBlockPattern.FindAllStringSubmatch(value, -1)
	if len(blocks) < 1 || len(blocks) > 8 {
		return ErrInvalid
	}
	for _, block := range blocks {
		der, err := base64.StdEncoding.DecodeString(lineBreaks.Replace(block[1]))
		if err != nil {
			return ErrInvalid
		}
		if _, err := x509.ParseCertificate(der); err != nil {
			return ErrInvalid
		}
	}
	return nil
}
