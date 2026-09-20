package scanplan

import (
	"encoding/json"
	"net"
	"net/url"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func scanURL(target string) (*url.URL, string) {
	request := contracts.PreparedHTTPRequest{Method: "GET", URL: target, Headers: []contracts.HTTPRequestHeader{}}
	if request.Validate() != nil || !(strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://")) {
		return nil, "unsupported_url"
	}
	u, err := url.Parse(target)
	if err != nil || !scanHost(u.Hostname()) {
		return nil, "unsupported_url"
	}
	return u, ""
}

func scanHost(host string) bool {
	if net.ParseIP(host) != nil {
		return true
	}
	if len(host) == 0 || len(host) > 253 {
		return false
	}
	for _, label := range strings.Split(strings.TrimSuffix(host, "."), ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, char := range label {
			if !(char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '-') {
				return false
			}
		}
	}
	return true
}

// Match the existing request-file adapter's representation limits before
// selecting a job. Test parameter names remain an explicit policy input.
func prepareSQLMapRequest(request contracts.PreparedHTTPRequest, names []string) (SQLMapRequest, string) {
	empty := SQLMapRequest{}
	u, code := scanURL(request.URL)
	if code != "" || request.Validate() != nil {
		return empty, "unsupported_request"
	}
	if strings.Contains(request.URL, "*") || strings.Contains(request.Body, "*") || strings.HasSuffix(request.Body, "\n") {
		return empty, "unsupported_request_representation"
	}
	if request.Body != "" {
		lines := strings.Split(request.Body, "\n")
		if strings.TrimSpace(lines[len(lines)-1]) == "" {
			return empty, "unsupported_request_representation"
		}
	}
	for _, char := range request.Body {
		if char < 32 && char != '\t' && char != '\n' {
			return empty, "unsupported_request_representation"
		}
	}
	query, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return empty, "unsupported_request_representation"
	}
	available := map[string]bool{}
	for name := range query {
		available[name] = true
	}
	contentType := ""
	bytesInHeaders := 0
	hasHost, hasLength := false, false
	for _, header := range request.Headers {
		name, value := strings.ToLower(header.Name), header.Value
		if strings.TrimSpace(value) != value {
			return empty, "unsupported_request_representation"
		}
		for _, char := range value {
			if char < 32 || char > 126 {
				return empty, "unsupported_request_representation"
			}
		}
		switch name {
		case "connection", "content-encoding", "expect", "if-modified-since", "if-none-match", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade":
			return empty, "unsupported_request_representation"
		case "host":
			hasHost = true
			if !strings.EqualFold(value, u.Host) {
				return empty, "unsupported_request_representation"
			}
		case "content-length":
			hasLength = true
			if value != strconv.Itoa(len(request.Body)) {
				return empty, "unsupported_request_representation"
			}
		case "content-type":
			contentType = value
			charsets := 0
			for _, parameter := range strings.Split(value, ";")[1:] {
				pair := strings.SplitN(parameter, "=", 2)
				if strings.ToLower(strings.TrimSpace(pair[0])) == "charset" {
					charsets++
					if len(pair) != 2 {
						return empty, "unsupported_request_representation"
					}
					charset := strings.ToLower(strings.Trim(strings.TrimSpace(pair[1]), "\""))
					if charset != "utf-8" && charset != "utf8" {
						return empty, "unsupported_request_representation"
					}
				}
			}
			if charsets > 1 {
				return empty, "unsupported_request_representation"
			}
		case "cookie":
			for _, cookie := range strings.Split(value, ";") {
				pair := strings.SplitN(strings.TrimSpace(cookie), "=", 2)
				if len(pair) == 2 {
					available[pair[0]] = true
				}
			}
		}
		markerValue := value
		if name == "accept" {
			markerValue = strings.ReplaceAll(value, "*/*", "")
		}
		if strings.Contains(header.Name, "*") || strings.Contains(markerValue, "*") {
			return empty, "unsupported_request_representation"
		}
		for _, selected := range names {
			if strings.EqualFold(selected, name) {
				available[selected] = true
			}
		}
		bytesInHeaders += len(header.Name) + len(value) + 4
	}
	if !hasHost {
		bytesInHeaders += len("Host: \r\n") + len(u.Host)
	}
	if !hasLength {
		bytesInHeaders += len("Content-Length: \r\n") + len(strconv.Itoa(len(request.Body)))
	}
	if bytesInHeaders > contracts.MaxHTTPRequestHeaderBytes {
		return empty, "unsupported_request_representation"
	}
	if strings.HasPrefix(strings.ToLower(contentType), "application/x-www-form-urlencoded") {
		if form, err := url.ParseQuery(request.Body); err == nil {
			for name := range form {
				available[name] = true
			}
		}
	} else if strings.HasPrefix(strings.ToLower(contentType), "application/json") {
		var body map[string]json.RawMessage
		if json.Unmarshal([]byte(request.Body), &body) == nil {
			for name := range body {
				available[name] = true
			}
		}
	}
	for _, name := range names {
		if !available[name] {
			return empty, "test_parameter_unavailable"
		}
	}
	raw := request.Method + " " + request.URL + " HTTP/1.1\r\n"
	for _, header := range request.Headers {
		raw += header.Name + ": " + header.Value + "\r\n"
	}
	raw += "\r\n" + request.Body
	lower := strings.ToLower(raw)
	for _, marker := range []string{"==========", "### conversation", "<request base64=", "%injecthere%", "%inject_here%", "%inject here%"} {
		if strings.Contains(lower, marker) {
			return empty, "unsupported_request_representation"
		}
	}
	return SQLMapRequest{SchemaVersion: 1, Method: request.Method, URL: request.URL, Headers: append([]contracts.HTTPRequestHeader{}, request.Headers...), Body: request.Body, TestParameters: append([]string{}, names...)}, ""
}
