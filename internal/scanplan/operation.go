package scanplan

import (
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var variablePattern = regexp.MustCompile(`\{([^{}]+)\}`)

func (p *preparer) prepareOperation(path, method string, item, op map[string]any, pointer string, urlOnly bool) (contracts.PreparedHTTPRequest, string) {
	empty := contracts.PreparedHTTPRequest{}
	server, code := p.server(item, op)
	if code != "" {
		return empty, code
	}
	if strings.ContainsAny(path, "?#\\") {
		return empty, "invalid_path"
	}
	parameters, code := p.parameters(item, op)
	if code != "" {
		return empty, code
	}
	input := p.options.Operations[pointer]
	used := map[string]bool{}
	headers := map[string]string{}
	query := url.Values{}
	cookies := map[string]string{}
	for _, key := range keys(parameters) {
		param := parameters[key]
		name := param["name"].(string)
		location := param["in"].(string)
		if urlOnly && location != "path" && location != "query" {
			p.gap(pointer, "non_url_parameter_not_applied")
			continue
		}
		if location == "header" && (strings.EqualFold(name, "accept") || strings.EqualFold(name, "content-type") || strings.EqualFold(name, "authorization")) {
			p.gap(pointer, "ignored_reserved_header_parameter")
			continue
		}
		required, code := booleanField(param, "required")
		if code != "" {
			return empty, code
		}
		// A path placeholder always needs data, regardless of schema metadata.
		required = required || location == "path"
		value, supplied := input.Parameters[key]
		if supplied {
			used[key] = true
		}
		text, exists, code := p.parameterValue(param, value, supplied)
		if code != "" {
			if required || supplied {
				return empty, code
			}
			p.gap(pointer, code)
			p.gap(pointer, "optional_parameter_omitted")
			continue
		}
		if !exists {
			if required {
				return empty, "missing_required_parameter"
			}
			p.gap(pointer, "optional_parameter_omitted")
			continue
		}
		switch location {
		case "path":
			marker := "{" + name + "}"
			if !strings.Contains(path, marker) {
				return empty, "invalid_path_parameter"
			}
			path = strings.ReplaceAll(path, marker, percentEncode(text))
		case "query":
			query.Set(name, text)
		case "header":
			headers[strings.ToLower(name)] = text
		case "cookie":
			cookies[name] = text
		}
	}
	for key := range input.Parameters {
		if !used[key] {
			return empty, "unknown_parameter_binding"
		}
	}
	if strings.ContainsAny(path, "{}") {
		return empty, "missing_path_parameter"
	}
	if urlOnly {
		if input.Body != nil || len(p.options.Authentication) != 0 {
			return empty, "unsupported_url_target_binding"
		}
		// Nuclei's URL interface runs its pinned templates, not this operation's
		// method/body or authenticated request. Never copy credentials into it.
		if method != "GET" {
			p.gap(pointer, "http_method_not_replayed")
		}
		if _, exists := op["requestBody"]; exists {
			p.gap(pointer, "request_body_not_replayed")
		}
		security, exists := op["security"]
		if !exists {
			security = p.root["security"]
		}
		if security != nil {
			if requirements, ok := security.([]any); !ok || len(requirements) != 0 {
				p.gap(pointer, "authentication_not_applied")
			}
		}
		return preparedRequestURL(server, path, query, "GET", map[string]string{}, "")
	}
	body, mediaType, code := p.body(op, input.Body, pointer)
	if code != "" {
		return empty, code
	}
	if mediaType != "" {
		headers["content-type"] = mediaType
	}
	if code = p.authenticate(op, headers, query, cookies); code != "" {
		return empty, code
	}
	if len(cookies) > 0 {
		if _, exists := headers["cookie"]; exists {
			return empty, "authentication_collision"
		}
		values := []string{}
		for _, name := range keys(cookies) {
			values = append(values, name+"="+cookies[name])
		}
		headers["cookie"] = strings.Join(values, "; ")
	}
	return preparedRequestURL(server, path, query, method, headers, body)
}

func preparedRequestURL(server, path string, query url.Values, method string, headers map[string]string, body string) (contracts.PreparedHTTPRequest, string) {
	request := contracts.PreparedHTTPRequest{Method: method, URL: strings.TrimSuffix(server, "/") + path, Headers: []contracts.HTTPRequestHeader{}, Body: body}
	if len(query) > 0 {
		request.URL += "?" + strings.ReplaceAll(query.Encode(), "+", "%20")
	}
	for _, name := range keys(headers) {
		request.Headers = append(request.Headers, contracts.HTTPRequestHeader{Name: name, Value: headers[name]})
	}
	if request.Validate() != nil {
		return contracts.PreparedHTTPRequest{}, "invalid_prepared_request"
	}
	return request, ""
}

func (p *preparer) server(item, op map[string]any) (string, string) {
	server := p.options.Server
	if server == "" {
		var raw any
		for _, object := range []map[string]any{op, item, p.root} {
			if value, exists := object["servers"]; exists {
				raw = value
				break
			}
		}
		servers, ok := raw.([]any)
		if !ok || len(servers) == 0 {
			return "", "missing_server"
		}
		selected, ok := servers[0].(map[string]any)
		if !ok {
			return "", "invalid_server"
		}
		server, ok = selected["url"].(string)
		if !ok {
			return "", "invalid_server"
		}
		variables := map[string]any{}
		if raw, exists := selected["variables"]; exists {
			variables, ok = raw.(map[string]any)
			if !ok {
				return "", "invalid_server"
			}
		}
		for _, match := range variablePattern.FindAllStringSubmatch(server, -1) {
			value, exists := p.options.ServerVariables[match[1]]
			if !exists {
				definition, _ := variables[match[1]].(map[string]any)
				value, ok = definition["default"].(string)
				if !ok {
					return "", "missing_server_variable"
				}
			}
			if strings.ContainsAny(value, "{}") {
				return "", "invalid_server_variable"
			}
			server = strings.ReplaceAll(server, match[0], value)
		}
	}
	if strings.ContainsAny(server, "{}?#") {
		return "", "invalid_server"
	}
	check := contracts.PreparedHTTPRequest{Method: "GET", URL: server, Headers: []contracts.HTTPRequestHeader{}, Body: ""}
	if check.Validate() != nil {
		return "", "invalid_server"
	}
	return server, ""
}

func (p *preparer) parameters(item, op map[string]any) (map[string]map[string]any, string) {
	result := map[string]map[string]any{}
	for _, object := range []map[string]any{item, op} {
		raw, exists := object["parameters"]
		if !exists {
			continue
		}
		values, ok := raw.([]any)
		if !ok || len(values) > 64 {
			return nil, "invalid_parameters"
		}
		seen := map[string]bool{}
		for _, value := range values {
			param, code := p.resolve(value, nil)
			if code != "" {
				return nil, code
			}
			name, ok := param["name"].(string)
			if !ok || name == "" || len(name) > 128 {
				return nil, "invalid_parameter"
			}
			location, ok := param["in"].(string)
			if !ok {
				return nil, "invalid_parameter"
			}
			switch location {
			case "path", "query", "header", "cookie":
			default:
				return nil, "unsupported_parameter_location"
			}
			key := location + ":" + name
			if location == "header" {
				key = strings.ToLower(key)
			}
			if seen[key] {
				return nil, "duplicate_parameter"
			}
			seen[key] = true
			result[key] = param
		}
	}
	if len(result) > 64 {
		return nil, "parameter_limit_exceeded"
	}
	return result, ""
}

// parameterValue validates only the serialization we can perform. Schema
// constraints do not determine whether concrete request data can be retained.
func (p *preparer) parameterValue(param map[string]any, value any, supplied bool) (string, bool, string) {
	exists := supplied
	if !supplied {
		var code string
		value, exists, code = p.example(param, param["schema"])
		if code != "" {
			return "", false, code
		}
	}
	if !exists {
		return "", false, ""
	}
	style := "form"
	if param["in"] == "path" || param["in"] == "header" {
		style = "simple"
	}
	if specified, exists := param["style"]; exists && specified != style {
		return "", false, "unsupported_parameter_style"
	}
	if _, exists := param["content"]; exists {
		return "", false, "unsupported_parameter_content"
	}
	if value, exists := param["allowReserved"]; exists && value != false {
		return "", false, "unsupported_allow_reserved"
	}
	if value, exists := param["explode"]; exists {
		if _, ok := value.(bool); !ok {
			return "", false, "invalid_parameter"
		}
	}
	text, ok := scalar(value)
	if !ok {
		return "", false, "unsupported_parameter_value"
	}
	name, _ := param["name"].(string)
	if param["in"] == "cookie" && (!cookieToken(name) || !cookieValue(text)) {
		return "", false, "invalid_cookie_parameter"
	}
	if param["in"] == "header" {
		check := contracts.PreparedHTTPRequest{Method: "GET", URL: "https://validation.invalid/", Headers: []contracts.HTTPRequestHeader{{Name: strings.ToLower(name), Value: text}}}
		if check.Validate() != nil {
			return "", false, "invalid_header_parameter"
		}
	}
	return text, true, ""
}

func (p *preparer) body(op map[string]any, input *BodyInput, pointer string) (string, string, string) {
	raw, exists := op["requestBody"]
	if input != nil {
		// Caller-supplied data is already concrete. A broken definition should
		// not prevent serialization; keep its diagnostic alongside the request.
		if !exists {
			p.gap(pointer, "unexpected_body_binding")
		} else if definition, code := p.resolve(raw, nil); code != "" {
			p.gap(pointer, code)
		} else if content, ok := definition["content"].(map[string]any); !ok {
			p.gap(pointer, "invalid_body")
		} else if _, declared := content[input.MediaType]; !declared {
			p.gap(pointer, "undeclared_body_media_type")
		}
		return serializeBody(input.Value, input.MediaType)
	}
	if !exists {
		return "", "", ""
	}
	body, code := p.resolve(raw, nil)
	if code != "" {
		return "", "", code
	}
	required, code := booleanField(body, "required")
	if code != "" {
		return "", "", code
	}
	omitOrFail := func(code string) (string, string, string) {
		if required {
			return "", "", code
		}
		p.gap(pointer, code)
		p.gap(pointer, "optional_body_omitted")
		return "", "", ""
	}
	content, ok := body["content"].(map[string]any)
	if !ok || len(content) == 0 {
		return omitOrFail("invalid_body")
	}
	lastCode := "unsupported_body_media_type"
	for _, mediaType := range keys(content) {
		if mediaType != "application/json" && mediaType != "text/plain" {
			continue
		}
		media, ok := content[mediaType].(map[string]any)
		if !ok {
			lastCode = "invalid_body"
			continue
		}
		value, exists, code := p.example(media, media["schema"])
		if code != "" {
			lastCode = code
			continue
		}
		if !exists {
			lastCode = "missing_required_body"
			continue
		}
		encoded, selectedType, code := serializeBody(value, mediaType)
		if code == "" {
			return encoded, selectedType, ""
		}
		lastCode = code
	}
	if !required && lastCode == "missing_required_body" {
		p.gap(pointer, "optional_body_omitted")
		return "", "", ""
	}
	return omitOrFail(lastCode)
}

func serializeBody(value any, mediaType string) (string, string, string) {
	if mediaType == "text/plain" {
		text, ok := value.(string)
		if !ok || len(text) > contracts.MaxHTTPRequestBodyBytes {
			return "", "", "invalid_body_value"
		}
		return text, mediaType, ""
	}
	if mediaType != "application/json" {
		return "", "", "unsupported_body_media_type"
	}
	encoded, err := contracts.MarshalPrivateCanonical(value)
	if err != nil || len(encoded) > contracts.MaxHTTPRequestBodyBytes {
		return "", "", "invalid_body_value"
	}
	return string(encoded), mediaType, ""
}

func booleanField(object map[string]any, key string) (bool, string) {
	value, exists := object[key]
	if !exists {
		return false, ""
	}
	result, ok := value.(bool)
	if !ok {
		return false, "invalid_boolean_field"
	}
	return result, ""
}
func scalar(value any) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, true
	case bool:
		return strconv.FormatBool(v), true
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), true
	default:
		return "", false
	}
}
func percentEncode(value string) string {
	return strings.ReplaceAll(url.QueryEscape(value), "+", "%20")
}
func cookieToken(value string) bool {
	if value == "" {
		return false
	}
	for _, c := range value {
		if c <= 32 || c >= 127 || strings.ContainsRune("()<>@,;:\\\"/[]?={}", c) {
			return false
		}
	}
	return true
}
func cookieValue(value string) bool {
	for _, c := range value {
		if c < 33 || c > 126 || strings.ContainsRune("\";,\\", c) {
			return false
		}
	}
	return true
}
