package scanplan

import (
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var bearerToken = regexp.MustCompile(`^[A-Za-z0-9._~+/-]+=*$`)

func (p *preparer) authenticate(op map[string]any, headers map[string]string, query url.Values, cookies map[string]string) string {
	raw, exists := op["security"]
	if !exists {
		raw, exists = p.root["security"]
	}
	if !exists {
		return ""
	}
	alternatives, ok := raw.([]any)
	if !ok || len(alternatives) > 64 {
		return "invalid_security"
	}
	if len(alternatives) == 0 {
		return ""
	}
	components, _ := p.root["components"].(map[string]any)
	schemes, _ := components["securitySchemes"].(map[string]any)
	reason := "missing_authentication"
	for _, alternative := range alternatives {
		requirement, ok := alternative.(map[string]any)
		if !ok || len(requirement) > 64 {
			return "invalid_security"
		}
		h := cloneMap(headers)
		c := cloneMap(cookies)
		q := url.Values{}
		for key, value := range query {
			q[key] = append([]string{}, value...)
		}
		valid := true
		for _, name := range keys(requirement) {
			scopes, ok := requirement[name].([]any)
			if !ok {
				return "invalid_security"
			}
			scheme, code := p.resolve(schemes[name], nil)
			if code != "" {
				reason = code
				valid = false
				break
			}
			kind, _ := scheme["type"].(string)
			httpScheme, _ := scheme["scheme"].(string)
			if kind != "apiKey" && !(kind == "http" && strings.EqualFold(httpScheme, "bearer")) {
				reason = "unsupported_authentication"
				valid = false
				break
			}
			if len(scopes) != 0 {
				return "invalid_security"
			}
			binding, exists := p.options.Authentication[name]
			value := binding.Reveal()
			if !exists || value == "" {
				reason = "missing_authentication"
				valid = false
				break
			}
			if len(value) > 8192 || strings.IndexFunc(value, func(c rune) bool { return c < 32 || c == 127 }) >= 0 {
				reason = "invalid_authentication"
				valid = false
				break
			}
			location := "header"
			field := "authorization"
			if kind == "http" {
				if !bearerToken.MatchString(value) {
					reason = "invalid_authentication"
					valid = false
					break
				}
				value = "Bearer " + value
			} else {
				field, ok = scheme["name"].(string)
				if !ok || field == "" {
					reason = "invalid_security"
					valid = false
					break
				}
				location, ok = scheme["in"].(string)
				if !ok {
					reason = "invalid_security"
					valid = false
					break
				}
			}
			switch location {
			case "header":
				field = strings.ToLower(field)
				if _, exists := h[field]; exists {
					reason = "authentication_collision"
					valid = false
					break
				}
				h[field] = value
			case "query":
				if _, exists := q[field]; exists {
					reason = "authentication_collision"
					valid = false
					break
				}
				q.Set(field, value)
			case "cookie":
				if !cookieToken(field) || !cookieValue(value) {
					reason = "invalid_authentication"
					valid = false
					break
				}
				if _, exists := c[field]; exists {
					reason = "authentication_collision"
					valid = false
					break
				}
				c[field] = value
			default:
				reason = "invalid_security"
				valid = false
			}
			if !valid {
				break
			}
		}
		if valid {
			if _, exists := h["cookie"]; exists && len(c) > 0 {
				reason = "authentication_collision"
				continue
			}
			check := contracts.PreparedHTTPRequest{Method: "GET", URL: "https://validation.invalid/", Headers: []contracts.HTTPRequestHeader{}, Body: ""}
			for _, name := range keys(h) {
				check.Headers = append(check.Headers, contracts.HTTPRequestHeader{Name: name, Value: h[name]})
			}
			if check.Validate() != nil {
				reason = "invalid_authentication"
				continue
			}
			for key, value := range h {
				headers[key] = value
			}
			for key, value := range q {
				query[key] = value
			}
			for key, value := range c {
				cookies[key] = value
			}
			return ""
		}
	}
	return reason
}

func cloneMap(value map[string]string) map[string]string {
	result := map[string]string{}
	for key, value := range value {
		result[key] = value
	}
	return result
}
