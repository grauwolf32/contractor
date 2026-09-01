package agentskills

import "strings"

func validateMemberPath(raw string) (string, bool, error) {
	if raw == "" || len(raw) > MaximumPathBytes || !isASCII(raw) || strings.Contains(raw, "\\") || strings.HasPrefix(raw, "/") {
		return "", false, validationError(CodePathInvalid, "")
	}
	directory := strings.HasSuffix(raw, "/")
	name := strings.TrimSuffix(raw, "/")
	if name == "" {
		return "", false, validationError(CodePathInvalid, "")
	}
	parts := strings.Split(name, "/")
	if len(parts) > MaximumPathComponents {
		return "", false, validationError(CodePathInvalid, "")
	}
	for index, part := range parts {
		if part == "" || part == "." || part == ".." {
			return "", false, validationError(CodePathInvalid, "")
		}
		if index == 0 && len(parts) == 1 && part == "SKILL.md" {
			continue
		}
		if !portableComponent(part) {
			return "", false, validationError(CodePathInvalid, "")
		}
	}
	return name, directory, nil
}

func portableComponent(value string) bool {
	if len(value) == 0 || len(value) > 128 || value[0] < 'a' || value[0] > 'z' && (value[0] < '0' || value[0] > '9') {
		return false
	}
	for _, char := range []byte(value[1:]) {
		if char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '.' || char == '_' || char == '-' {
			continue
		}
		return false
	}
	return true
}

func isASCII(value string) bool {
	for _, char := range []byte(value) {
		if char > 0x7f {
			return false
		}
	}
	return true
}

func validSkillName(value string) bool {
	if len(value) < 1 || len(value) > 64 || value[0] == '-' || value[len(value)-1] == '-' {
		return false
	}
	previousHyphen := false
	for _, char := range []byte(value) {
		if char == '-' {
			if previousHyphen {
				return false
			}
			previousHyphen = true
			continue
		}
		previousHyphen = false
		if (char < 'a' || char > 'z') && (char < '0' || char > '9') {
			return false
		}
	}
	return true
}
