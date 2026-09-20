package scanplan

import (
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var variablePattern = regexp.MustCompile(`\{([^{}]+)\}`)

func (p *preparer) operation(path, method string, item, op map[string]any, pointer string) (contracts.PreparedHTTPRequest, string) {
	empty := contracts.PreparedHTTPRequest{}
	if _, exists := op["requestBody"]; exists && (method == "GET" || method == "HEAD" || method == "DELETE") {
		return empty, "unsupported_body_method"
	}
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
		if location == "header" && (strings.EqualFold(name, "accept") || strings.EqualFold(name, "content-type") || strings.EqualFold(name, "authorization")) {
			p.gap(pointer, "ignored_reserved_header_parameter")
			continue
		}
		style := "form"
		if location == "path" || location == "header" {
			style = "simple"
		}
		if specified, exists := param["style"]; exists && specified != style {
			return empty, "unsupported_parameter_style"
		}
		if _, exists := param["content"]; exists {
			return empty, "unsupported_parameter_content"
		}
		if value, exists := param["allowReserved"]; exists && value != false {
			return empty, "unsupported_allow_reserved"
		}
		if value, exists := param["explode"]; exists {
			if _, ok := value.(bool); !ok {
				return empty, "invalid_parameter"
			}
		}
		required, code := booleanField(param, "required")
		if code != "" {
			return empty, code
		}
		if location == "path" && !required {
			return empty, "invalid_path_parameter"
		}
		schema, code := p.schema(param["schema"], nil, 1)
		if code != "" {
			return empty, code
		}
		value, exists := input.Parameters[key]
		if exists {
			used[key] = true
		} else {
			value, exists, code = p.example(param, schema)
			if code != "" {
				return empty, code
			}
		}
		if !exists {
			if required {
				return empty, "missing_required_parameter"
			}
			p.gap(pointer, "optional_parameter_omitted")
			continue
		}
		if code = validateExample(value, schema); code != "" {
			return empty, code
		}
		text, ok := scalar(value)
		if !ok {
			return empty, "unsupported_parameter_value"
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
			if !cookieToken(name) || !cookieValue(text) {
				return empty, "invalid_cookie_parameter"
			}
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
	request := contracts.PreparedHTTPRequest{Method: method, URL: strings.TrimSuffix(server, "/") + path, Headers: []contracts.HTTPRequestHeader{}, Body: body}
	if len(query) > 0 {
		request.URL += "?" + strings.ReplaceAll(query.Encode(), "+", "%20")
	}
	for _, name := range keys(headers) {
		request.Headers = append(request.Headers, contracts.HTTPRequestHeader{Name: name, Value: headers[name]})
	}
	if request.Validate() != nil {
		return empty, "invalid_prepared_request"
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
			definition, ok := variables[match[1]].(map[string]any)
			if !ok {
				return "", "missing_server_variable"
			}
			value, exists := p.options.ServerVariables[match[1]]
			if !exists {
				value, ok = definition["default"].(string)
				if !ok {
					return "", "missing_server_variable"
				}
			}
			if strings.ContainsAny(value, "{}") {
				return "", "invalid_server_variable"
			}
			if raw, exists := definition["enum"]; exists {
				values, ok := raw.([]any)
				if !ok {
					return "", "invalid_server_variable"
				}
				found := false
				for _, allowed := range values {
					if text, ok := allowed.(string); ok && text == value {
						found = true
					}
				}
				if !found {
					return "", "invalid_server_variable"
				}
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

func (p *preparer) body(op map[string]any, input *BodyInput, pointer string) (string, string, string) {
	raw, exists := op["requestBody"]
	if !exists {
		if input != nil {
			return "", "", "unexpected_body_binding"
		}
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
	content, ok := body["content"].(map[string]any)
	if !ok || len(content) == 0 {
		return "", "", "invalid_body"
	}
	mediaType := ""
	if input != nil {
		mediaType = input.MediaType
	} else {
		for _, candidate := range keys(content) {
			if candidate == "application/json" || candidate == "text/plain" {
				mediaType = candidate
				break
			}
		}
	}
	if mediaType != "application/json" && mediaType != "text/plain" {
		return "", "", "unsupported_body_media_type"
	}
	media, ok := content[mediaType].(map[string]any)
	if !ok {
		return "", "", "undeclared_body_media_type"
	}
	if _, exists := media["encoding"]; exists {
		return "", "", "unsupported_body_encoding"
	}
	schema := map[string]any{}
	if raw, exists := media["schema"]; exists {
		schema, code = p.schema(raw, nil, 1)
		if code != "" {
			return "", "", code
		}
	}
	var value any
	if input != nil {
		value = input.Value
		exists = true
	} else {
		value, exists, code = p.example(media, schema)
		if code != "" {
			return "", "", code
		}
	}
	if !exists {
		if required {
			return "", "", "missing_required_body"
		}
		p.gap(pointer, "optional_body_omitted")
		return "", "", ""
	}
	if code = validateExample(value, schema); code != "" {
		return "", "", code
	}
	if mediaType == "text/plain" {
		text, ok := value.(string)
		if !ok {
			return "", "", "invalid_body_value"
		}
		return text, mediaType, ""
	}
	encoded, err := contracts.MarshalPrivateCanonical(value)
	if err != nil {
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
