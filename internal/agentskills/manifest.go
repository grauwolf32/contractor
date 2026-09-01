package agentskills

import (
	"bytes"
	"io"
	"strings"
	"unicode/utf8"

	"go.yaml.in/yaml/v4"
)

func parseManifest(data []byte, expectedName string) (Manifest, error) {
	if len(data) > MaximumManifestBytes {
		return Manifest{}, validationError(CodeLimitExceeded, "SKILL.md")
	}
	if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
		return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
	}
	frontmatter, body, ok := splitFrontmatter(data)
	if !ok || len(frontmatter) > MaximumFrontmatterBytes || len(bytes.TrimSpace(body)) == 0 {
		code := CodeManifestInvalid
		if len(frontmatter) > MaximumFrontmatterBytes {
			code = CodeLimitExceeded
		}
		return Manifest{}, validationError(code, "SKILL.md")
	}

	decoder := yaml.NewDecoder(bytes.NewReader(frontmatter))
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
	}
	var trailing yaml.Node
	if err := decoder.Decode(&trailing); err != io.EOF {
		return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
	}
	root, err := validateYAMLTree(&document)
	if err != nil {
		return Manifest{}, err
	}
	result, err := constructManifest(root)
	if err != nil {
		return Manifest{}, err
	}
	if expectedName != "" && result.Name != expectedName {
		return Manifest{}, validationError(CodeNameMismatch, "SKILL.md")
	}
	return result, nil
}

func splitFrontmatter(data []byte) ([]byte, []byte, bool) {
	if !bytes.HasPrefix(data, []byte("---\n")) {
		return nil, nil, false
	}
	rest := data[4:]
	if index := bytes.Index(rest, []byte("\n---\n")); index >= 0 {
		return rest[:index], rest[index+5:], true
	}
	return nil, nil, false
}

func validateYAMLTree(document *yaml.Node) (*yaml.Node, error) {
	if document == nil || document.Kind != yaml.DocumentNode || len(document.Content) != 1 {
		return nil, validationError(CodeManifestInvalid, "SKILL.md")
	}
	nodes := 0
	var walk func(*yaml.Node, int) error
	walk = func(node *yaml.Node, depth int) error {
		nodes++
		if nodes > MaximumYAMLNodes || depth > MaximumYAMLDepth {
			return validationError(CodeLimitExceeded, "SKILL.md")
		}
		if node.Alias != nil || node.Anchor != "" || node.Kind == yaml.AliasNode {
			return validationError(CodeManifestInvalid, "SKILL.md")
		}
		switch node.Kind {
		case yaml.MappingNode:
			if node.Tag != "!!map" && node.Tag != "tag:yaml.org,2002:map" {
				return validationError(CodeManifestInvalid, "SKILL.md")
			}
		case yaml.ScalarNode:
			if node.Tag != "!!str" && node.Tag != "tag:yaml.org,2002:str" {
				return validationError(CodeManifestInvalid, "SKILL.md")
			}
		default:
			return validationError(CodeManifestInvalid, "SKILL.md")
		}
		for _, child := range node.Content {
			if err := walk(child, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	root := document.Content[0]
	if err := walk(root, 1); err != nil {
		return nil, err
	}
	if root.Kind != yaml.MappingNode || len(root.Content)%2 != 0 {
		return nil, validationError(CodeManifestInvalid, "SKILL.md")
	}
	return root, nil
}

func constructManifest(root *yaml.Node) (Manifest, error) {
	values := make(map[string]*yaml.Node, len(root.Content)/2)
	for index := 0; index < len(root.Content); index += 2 {
		key := root.Content[index]
		value := root.Content[index+1]
		if key.Kind != yaml.ScalarNode || !isStringTag(key.Tag) {
			return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
		}
		if _, exists := values[key.Value]; exists {
			return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
		}
		switch key.Value {
		case "name", "description", "license", "compatibility", "metadata":
		default:
			return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
		}
		values[key.Value] = value
	}
	name, ok := scalarString(values["name"])
	if !ok {
		return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
	}
	if len(name) > 64 {
		return Manifest{}, validationError(CodeLimitExceeded, "SKILL.md")
	}
	if !validSkillName(name) {
		return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
	}
	description, ok := scalarString(values["description"])
	if !ok || strings.TrimSpace(description) == "" {
		return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
	}
	if len([]byte(description)) > MaximumDescriptionBytes {
		return Manifest{}, validationError(CodeLimitExceeded, "SKILL.md")
	}
	result := Manifest{Name: name, Description: description}
	var valid bool
	if node, exists := values["license"]; exists {
		result.License, valid = scalarString(node)
		if !valid {
			return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
		}
		if len([]byte(result.License)) > MaximumLicenseBytes {
			return Manifest{}, validationError(CodeLimitExceeded, "SKILL.md")
		}
	}
	if node, exists := values["compatibility"]; exists {
		result.Compatibility, valid = scalarString(node)
		if !valid {
			return Manifest{}, validationError(CodeManifestInvalid, "SKILL.md")
		}
		if len([]byte(result.Compatibility)) > MaximumCompatibilityBytes {
			return Manifest{}, validationError(CodeLimitExceeded, "SKILL.md")
		}
	}
	if node, exists := values["metadata"]; exists {
		metadata, err := constructMetadata(node)
		if err != nil {
			return Manifest{}, err
		}
		result.Metadata = metadata
	}
	return result, nil
}

func constructMetadata(node *yaml.Node) (map[string]string, error) {
	if node == nil || node.Kind != yaml.MappingNode || !isMapTag(node.Tag) || len(node.Content)%2 != 0 {
		return nil, validationError(CodeManifestInvalid, "SKILL.md")
	}
	if len(node.Content)/2 > MaximumMetadataEntries {
		return nil, validationError(CodeLimitExceeded, "SKILL.md")
	}
	result := make(map[string]string, len(node.Content)/2)
	for index := 0; index < len(node.Content); index += 2 {
		keyNode, valueNode := node.Content[index], node.Content[index+1]
		key, keyOK := scalarString(keyNode)
		value, valueOK := scalarString(valueNode)
		if !keyOK || !valueOK || strings.HasPrefix(key, "adk_") {
			return nil, validationError(CodeManifestInvalid, "SKILL.md")
		}
		if len(key) > 64 || len([]byte(value)) > MaximumMetadataValueBytes {
			return nil, validationError(CodeLimitExceeded, "SKILL.md")
		}
		if !validMetadataKey(key) {
			return nil, validationError(CodeManifestInvalid, "SKILL.md")
		}
		if _, exists := result[key]; exists {
			return nil, validationError(CodeManifestInvalid, "SKILL.md")
		}
		result[key] = value
	}
	return result, nil
}

func scalarString(node *yaml.Node) (string, bool) {
	returnValue := ""
	if node == nil || node.Kind != yaml.ScalarNode || !isStringTag(node.Tag) {
		return returnValue, false
	}
	return node.Value, true
}

func isStringTag(tag string) bool { return tag == "!!str" || tag == "tag:yaml.org,2002:str" }
func isMapTag(tag string) bool    { return tag == "!!map" || tag == "tag:yaml.org,2002:map" }

func validMetadataKey(value string) bool {
	if len(value) < 1 || len(value) > 64 || !isASCII(value) {
		return false
	}
	for _, char := range []byte(value) {
		if char >= 'A' && char <= 'Z' || char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '_' || char == '.' || char == '-' {
			continue
		}
		return false
	}
	return true
}
