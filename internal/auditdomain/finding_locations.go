package auditdomain

import (
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type FindingLineRange struct {
	StartLine int64 `json:"start_line"`
	EndLine   int64 `json:"end_line"`
}

type FindingLocation struct {
	File   string            `json:"file,omitempty"`
	Line   *int64            `json:"line,omitempty"`
	Range  *FindingLineRange `json:"range,omitempty"`
	URL    string            `json:"url,omitempty"`
	Method string            `json:"method,omitempty"`
}

// RFC 9110 token characters, shared by methods and field names.
var httpToken = regexp.MustCompile("^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

func validateFindingExtensions(proposal FindingProposal) error {
	if len(proposal.Locations) > MaximumFindingLocations {
		return invalid(CodeLimitExceeded, "locations")
	}
	for _, location := range proposal.Locations {
		if err := validateFindingLocation(location); err != nil {
			return err
		}
	}
	if proposal.HTTPExchange != nil {
		return validateFindingHTTPExchange(*proposal.HTTPExchange, proposal.EvidenceIDs)
	}
	return nil
}

func validateFindingLocation(location FindingLocation) error {
	hasFile, hasURL := location.File != "", location.URL != ""
	if hasFile == hasURL {
		return invalid(CodeInvalid, "locations")
	}
	if hasFile {
		return validateFindingSourceLocation(location)
	}
	if !validFindingURL(location.URL) || location.Line != nil || location.Range != nil {
		return invalid(CodeInvalid, "locations.url")
	}
	if location.Method != "" && !validFindingMethod(location.Method) {
		return invalid(CodeInvalid, "locations.method")
	}
	return nil
}

func validateFindingSourceLocation(location FindingLocation) error {
	path := location.File
	if len(path) > MaximumFindingPathBytes || !utf8.ValidString(path) ||
		strings.ContainsAny(path, "\\:") || strings.IndexFunc(path, unicode.IsControl) >= 0 {
		return invalid(CodeInvalid, "locations.file")
	}
	for _, component := range strings.Split(path, "/") {
		if component == "" || component == "." || component == ".." {
			return invalid(CodeInvalid, "locations.file")
		}
	}
	if location.Method != "" || (location.Line != nil && location.Range != nil) {
		return invalid(CodeInvalid, "locations")
	}
	if location.Line != nil && !validFindingLine(*location.Line) {
		return invalid(CodeInvalid, "locations.line")
	}
	if region := location.Range; region != nil {
		if !validFindingLine(region.StartLine) || !validFindingLine(region.EndLine) || region.EndLine < region.StartLine {
			return invalid(CodeInvalid, "locations.range")
		}
	}
	return nil
}

func validFindingLine(line int64) bool {
	return line >= 1 && line <= MaximumFindingLine
}

func validFindingMethod(method string) bool {
	return len(method) <= MaximumFindingMethodBytes && httpToken.MatchString(method)
}

func validFindingURL(value string) bool {
	if len(value) > MaximumFindingURLBytes || !utf8.ValidString(value) || strings.Contains(value, "\\") {
		return false
	}
	if strings.IndexFunc(value, func(character rune) bool {
		return unicode.IsSpace(character) || unicode.IsControl(character)
	}) >= 0 || !validPercentEscapes(value) {
		return false
	}
	parsed, err := url.Parse(value)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.Hostname() == "" || parsed.User != nil || parsed.Opaque != "" {
		return false
	}
	if strings.HasSuffix(parsed.Host, ":") {
		return false
	}
	if port := parsed.Port(); port != "" {
		// A network port is an unsigned 16-bit number; zero is not a target service.
		number, err := strconv.ParseUint(port, 10, 16)
		if err != nil || number == 0 {
			return false
		}
	}
	return true
}

func validPercentEscapes(value string) bool {
	// A URI escape consists of '%' and two hex digits. Validate queries and
	// fragments too, without rewriting the authored URL.
	for index := 0; index < len(value); index++ {
		if value[index] != '%' {
			continue
		}
		if index+2 >= len(value) {
			return false
		}
		if _, err := strconv.ParseUint(value[index+1:index+3], 16, 8); err != nil {
			return false
		}
		index += 2
	}
	return true
}
