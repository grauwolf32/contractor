package public

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/config"
	"golang.org/x/text/cases"
	"golang.org/x/text/unicode/norm"
)

const maxCatalogSearchRunes = 200

type catalogQuery struct {
	Search string
	Name   string
}

func parseCatalogQuery(search, name string, namePresent bool) (catalogQuery, error) {
	if !utf8.ValidString(search) {
		return catalogQuery{}, fmt.Errorf("%w: q must be valid UTF-8", errInvalidRequest)
	}
	if utf8.RuneCountInString(search) > maxCatalogSearchRunes {
		return catalogQuery{}, fmt.Errorf(
			"%w: q must contain at most %d Unicode characters",
			errInvalidRequest, maxCatalogSearchRunes,
		)
	}
	normalizedSearch := norm.NFC.String(strings.TrimSpace(search))
	normalizedSearch = foldCatalogText(normalizedSearch)
	if namePresent {
		if _, err := config.ParseSelector(name + "@1"); err != nil {
			return catalogQuery{}, fmt.Errorf("%w: name must be an exact configuration name", errInvalidRequest)
		}
	}
	return catalogQuery{Search: normalizedSearch, Name: name}, nil
}

func catalogMatches(query catalogQuery, name, version string, authored ...string) bool {
	if query.Name != "" && name != query.Name {
		return false
	}
	if query.Search == "" {
		return true
	}
	fields := append([]string{name, version}, authored...)
	for _, field := range fields {
		folded := foldCatalogText(field)
		if strings.Contains(folded, query.Search) {
			return true
		}
	}
	return false
}

func foldCatalogText(value string) string {
	// Caser owns mutable transform state, so create one per operation instead
	// of sharing it between concurrent HTTP requests.
	return cases.Fold().String(norm.NFC.String(value))
}

func catalogPageCursorKind(base string, query catalogQuery, sourceFingerprint string) string {
	return base + "\x00q=" + query.Search + "\x00name=" + query.Name + "\x00source=" + sourceFingerprint
}

func configurationDescription(resource config.ConfigurationResource) string {
	body, ok := resource.Body.(map[string]any)
	if !ok {
		return ""
	}
	description, _ := body["description"].(string)
	return description
}
