package sarif

import (
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
)

type projectedLocation struct {
	value   object
	native  findinganalysis.Location
	localID int
}

func (r *registry) locations(receipt findingexport.Receipt) (map[string]projectedLocation, error) {
	result := map[string]projectedLocation{}
	if receipt.Analysis == nil {
		return result, nil
	}
	sources := map[string]findinganalysis.ResolvedSource{}
	for _, s := range receipt.Analysis.Sources {
		sources[s.ID] = s
	}
	native := map[string]findinganalysis.Location{}
	for _, l := range receipt.Analysis.Analysis.Locations {
		native[l.ID] = l
	}
	for i, id := range sortedKeys(native) {
		l := native[id]
		source, exists := sources[l.SourceID]
		if !exists {
			return nil, findingexport.ErrInvalid
		}
		index, exists := r.fileIndex[fileKey(source.RootKey, l.Path)]
		if !exists {
			return nil, findingexport.ErrInvalid
		}
		physical := object{"artifactLocation": object{"uri": relativeURI(l.Path), "uriBaseId": baseID(source.RootKey), "index": index}}
		if region := l.Region; region != nil {
			projected := object{"startLine": region.StartLine}
			if region.EndLine != nil {
				projected["endLine"] = *region.EndLine
			}
			if region.StartColumn != nil {
				projected["startColumn"] = *region.StartColumn
			}
			if region.EndColumn != nil {
				projected["endColumn"] = *region.EndColumn
			}
			physical["region"] = projected
		}
		value := object{"physicalLocation": physical, "properties": object{"contractor": object{"native_location_id": l.ID, "source_id": l.SourceID}}}
		if symbol := l.Symbol; symbol != nil {
			logical := object{"name": symbol.Name}
			if symbol.QualifiedName != "" {
				logical["fullyQualifiedName"] = symbol.QualifiedName
			}
			if symbol.Kind != "" {
				logical["kind"] = symbol.Kind
			}
			details := object{}
			if symbol.Language != "" {
				details["language"] = symbol.Language
			}
			if symbol.Signature != "" {
				details["signature"] = symbol.Signature
			}
			if len(details) > 0 {
				logical["properties"] = object{"contractor": details}
			}
			value["logicalLocations"] = []any{logical}
		}
		result[id] = projectedLocation{value: value, native: l, localID: i + 1}
	}
	return result, nil
}

func locationWithID(location projectedLocation) object {
	result := object{}
	for k, v := range location.value {
		result[k] = v
	}
	result["id"] = location.localID
	return result
}
