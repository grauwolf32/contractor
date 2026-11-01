package projectsitemap

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

var (
	ErrInvalid             = apierror.New(apierror.InvalidRequest, "invalid sitemap document or mutation")
	ErrConflict            = apierror.New(apierror.SitemapConflict, "sitemap edit or source precondition conflict")
	errStructuralSelection = errors.New("sitemap structural selection is unresolved")
)

// Decode validates stored bytes strictly. Missing bindings are handled by the
// service; invalid stored content is never interpreted as an empty map.
func Decode(data []byte) (Document, error) {
	if len(data) > MaxDocumentBytes {
		return Document{}, ErrInvalid
	}
	var document Document
	if err := strictjson.DecodeNonNull(data, &document); err != nil {
		return Document{}, ErrInvalid
	}
	if err := validateDocument(document); err != nil {
		return Document{}, err
	}
	return document, nil
}

func encode(document Document) ([]byte, error) {
	canonicalize(&document)
	if err := validateDocument(document); err != nil {
		return nil, err
	}
	data, err := json.Marshal(document)
	if err != nil || len(data) > MaxDocumentBytes {
		return nil, ErrInvalid
	}
	return data, nil
}

func canonicalize(d *Document) {
	for i := range d.Rules {
		if d.Rules[i].Parameters == nil {
			d.Rules[i].Parameters = []string{}
		}
		if d.Rules[i].Structural == nil {
			d.Rules[i].Structural = []StructuralParameter{}
		}
	}
	slices.SortFunc(d.Groups, func(a, b Group) int { return strings.Compare(a.ID, b.ID) })
	slices.SortFunc(d.Sources, func(a, b Source) int { return strings.Compare(a.Provenance.SourceID, b.Provenance.SourceID) })
	slices.SortFunc(d.Rules, func(a, b Rule) int { return bytes.Compare(jsonBytes(a), jsonBytes(b)) })
	slices.SortFunc(d.Markers, func(a, b Marker) int { return strings.Compare(a.ID, b.ID) })
	slices.SortFunc(d.Gaps, func(a, b Gap) int { return strings.Compare(a.SourceID+"/"+a.Reason, b.SourceID+"/"+b.Reason) })
	for i := range d.Groups {
		g := &d.Groups[i]
		if g.Members == nil {
			g.Members = []Member{}
		}
		if g.Declarations == nil {
			g.Declarations = []Declaration{}
		}
		for j := range g.Declarations {
			if g.Declarations[j].Parameters == nil {
				g.Declarations[j].Parameters = []string{}
			}
		}
		for j := range g.Structural {
			if g.Structural[j].Values == nil {
				g.Structural[j].Values = []string{}
			}
		}
		if g.Structural == nil {
			g.Structural = []StructuralValue{}
		}
		slices.SortFunc(g.Members, func(a, b Member) int { return strings.Compare(a.ID, b.ID) })
		slices.SortFunc(g.Declarations, func(a, b Declaration) int { return bytes.Compare(jsonBytes(a), jsonBytes(b)) })
		for j := range g.Members {
			m := &g.Members[j]
			if m.Summaries == nil {
				m.Summaries = []Summary{}
			}
			if m.Examples == nil {
				m.Examples = []string{}
			}
			if m.Evidence == nil {
				m.Evidence = []Evidence{}
			}
			if m.Annotations == nil {
				m.Annotations = []Annotation{}
			}
			for k := range m.Summaries {
				if m.Summaries[k].Parameters == nil {
					m.Summaries[k].Parameters = []string{}
				}
			}
			for k := range m.Annotations {
				if m.Annotations[k].Tags == nil {
					m.Annotations[k].Tags = []string{}
				}
			}
			slices.Sort(m.Examples)
			slices.SortFunc(m.Summaries, func(a, b Summary) int { return strings.Compare(summaryKey(a), summaryKey(b)) })
			slices.SortFunc(m.Annotations, func(a, b Annotation) int { return bytes.Compare(jsonBytes(a), jsonBytes(b)) })
			slices.SortFunc(m.Evidence, func(a, b Evidence) int { return bytes.Compare(jsonBytes(a), jsonBytes(b)) })
		}
	}
}

func jsonBytes(v any) []byte      { data, _ := json.Marshal(v); return data }
func summaryKey(s Summary) string { s.First, s.Last = "", ""; return string(jsonBytes(s)) }
func groupID(origin, method, route string, structural ...StructuralValue) string {
	if len(structural) == 0 {
		return key(origin, method, route)
	}
	return key(origin, method, route, string(jsonBytes(structural)))
}
func memberID(origin, method, path string, structural ...StructuralValue) string {
	if len(structural) == 0 {
		return key("member", origin, method, path)
	}
	return key("member", origin, method, path, string(jsonBytes(structural)))
}

func validateDocument(d Document) error {
	if d.Version != Version || d.Projection != Projection || d.Groups == nil || d.Sources == nil ||
		d.Rules == nil || d.Markers == nil || d.Gaps == nil || len(d.Groups) > MaxGroups ||
		len(d.Sources) > MaxSources || len(d.Rules) > MaxRules || len(d.Markers) > MaxMarkers || len(d.Gaps) > (MaxSources+1)*16 {
		return ErrInvalid
	}
	sources := map[string]Source{}
	for _, source := range d.Sources {
		if _, duplicate := sources[source.Provenance.SourceID]; duplicate || source.Generation == 0 ||
			!validProvenance(source.Provenance) || !hashPattern.MatchString(source.PolicyDigest) || source.Generation > MaxOmissionCount {
			return ErrInvalid
		}
		if source.Artifact != nil && source.Artifact.Validate() != nil {
			return ErrInvalid
		}
		sources[source.Provenance.SourceID] = source
	}
	for _, rule := range d.Rules {
		if rule.Parameters == nil || rule.Structural == nil {
			return ErrInvalid
		}
		if _, found := sources[rule.SourceID]; !found {
			return ErrInvalid
		}
		if validateRule(rule) != nil {
			return ErrInvalid
		}
	}
	groups, members := map[string]bool{}, map[string]bool{}
	for _, g := range d.Groups {
		if g.Structural == nil || g.Members == nil || g.Declarations == nil ||
			g.ID != groupID(g.Origin, g.Method, g.Route, g.Structural...) || groups[g.ID] || !methodPattern.MatchString(g.Method) || !validStructural(g.Structural) ||
			len(g.Members) > MaxGroupMembers || len(g.Declarations) > MaxRules || g.Omitted > MaxOmissionCount || !validRoute(g.Origin, g.Route) {
			return ErrInvalid
		}
		groups[g.ID] = true
		for _, value := range g.Structural {
			if value.Values == nil {
				return ErrInvalid
			}
		}
		examples := 0
		for _, m := range g.Members {
			if m.Summaries == nil || m.Examples == nil || m.Evidence == nil || m.Annotations == nil ||
				m.ID != memberID(g.Origin, g.Method, m.Path, g.Structural...) || members[m.ID] ||
				len(m.Summaries) > MaxSummaries || len(m.Evidence) > MaxProvenance || len(m.Annotations) > MaxSources ||
				len(m.Path) > MaxMemberKeyBytes || m.Omitted > MaxOmissionCount || !validRoute(g.Origin, m.Path) {
				return ErrInvalid
			}
			members[m.ID] = true
			seenSummaries := map[string]bool{}
			for _, s := range m.Summaries {
				if s.Parameters == nil || !observedKind(s.Provenance.Kind) || !admittedProvenance(s.Provenance, sources) || seenSummaries[summaryKey(s)] || !validOutcome(s.Outcome, s.Status) ||
					!validTime(s.First) || !validTime(s.Last) || timeBefore(s.Last, s.First) || !validNames(s.Parameters) {
					return ErrInvalid
				}
				seenSummaries[summaryKey(s)] = true
			}
			examples += len(m.Examples)
			for _, example := range m.Examples {
				if len(example) > MaxExampleBytes || example != g.Method+" "+g.Origin+m.Path {
					return ErrInvalid
				}
			}
			for _, evidence := range m.Evidence {
				if !contracts.ValidResourceID(evidence.RunID) || evidence.Artifact.Validate() != nil {
					return ErrInvalid
				}
			}
			for _, annotation := range m.Annotations {
				if annotation.Tags == nil || !admittedProvenance(annotation.Provenance, sources) || len(annotation.Note) > MaxNoteBytes ||
					!validText(annotation.Note) || len(annotation.Tags) > MaxTags || !validNames(annotation.Tags) {
					return ErrInvalid
				}
			}
		}
		if examples > MaxExamples {
			return ErrInvalid
		}
		for _, declaration := range g.Declarations {
			if declaration.Parameters == nil || !admittedProvenance(declaration.Provenance, sources) || !validNames(declaration.Parameters) || len(declaration.Pointer) > MaxPointerBytes ||
				!validText(declaration.Pointer) {
				return ErrInvalid
			}
		}
	}
	if len(members) > MaxMembers {
		return ErrInvalid
	}
	markerIDs := map[string]bool{}
	for _, marker := range d.Markers {
		if !contracts.ValidResourceID(marker.ID) || markerIDs[marker.ID] || len(marker.Members) == 0 || len(marker.Members) > MaxGroupMembers ||
			!admittedProvenance(marker.Provenance, sources) {
			return ErrInvalid
		}
		markerIDs[marker.ID] = true
		for _, id := range marker.Members {
			if !members[id] {
				return ErrInvalid
			}
		}
	}
	gapIDs := map[string]bool{}
	for _, gap := range d.Gaps {
		id := gap.SourceID + "/" + gap.Reason
		if gapIDs[id] || gap.Count > MaxOmissionCount || (gap.SourceID != "" && !contracts.ValidResourceID(gap.SourceID)) ||
			!validGapReason(gap.Reason) {
			return ErrInvalid
		}
		gapIDs[id] = true
	}
	return nil
}

func validProvenance(p Provenance) bool {
	return (Authority{SourceID: p.SourceID, Kind: p.Kind, Deployment: p.Deployment}).validate() == nil
}
func admittedProvenance(p Provenance, sources map[string]Source) bool {
	source, found := sources[p.SourceID]
	return found && validProvenance(p) && source.Provenance.Kind == p.Kind && source.Provenance.Deployment == p.Deployment
}
func validNames(names []string) bool {
	if len(names) > MaxBatch {
		return false
	}
	for _, name := range names {
		if !validLabel(name) {
			return false
		}
	}
	return true
}
func validTime(value string) bool {
	t, err := time.Parse(time.RFC3339Nano, value)
	return err == nil && t.UTC().Format(time.RFC3339Nano) == value
}
func timeBefore(a, b string) bool {
	first, _ := time.Parse(time.RFC3339Nano, a)
	second, _ := time.Parse(time.RFC3339Nano, b)
	return first.Before(second)
}
func validOutcome(outcome string, status int) bool {
	return (outcome == "response" && status >= 100 && status <= 599) ||
		((outcome == "timeout" || outcome == "transport-error" || outcome == "attempt") && status == 0)
}
func validRoute(origin, path string) bool {
	if !strings.HasPrefix(path, "/") || strings.ContainsAny(path, "?#") || len(origin)+len(path) > MaxMemberKeyBytes || !validText(path) {
		return false
	}
	o, _, _, err := projectURL(origin + path)
	return err == nil && o == origin
}
func validateRule(rule Rule) error {
	if !contracts.ValidResourceID(rule.SourceID) || !methodPattern.MatchString(rule.Method) || !validRoute(rule.Origin, rule.Path) ||
		!validNames(rule.Parameters) || len(rule.Pointer) > MaxPointerBytes || !validText(rule.Pointer) {
		return ErrInvalid
	}
	for _, segment := range strings.Split(rule.Path, "/") {
		if strings.ContainsAny(segment, "{}") && !(strings.HasPrefix(segment, "{") && strings.HasSuffix(segment, "}") && validLabel(segment[1:len(segment)-1])) {
			return ErrInvalid
		}
	}
	if len(rule.Structural) > MaxTags {
		return ErrInvalid
	}
	names := map[string]bool{}
	for _, selector := range rule.Structural {
		if names[selector.Name] || !validLabel(selector.Name) || len(selector.Allowed) == 0 || len(selector.Allowed) > MaxTags {
			return ErrInvalid
		}
		names[selector.Name] = true
		for _, value := range selector.Allowed {
			if len(value) > MaxTagBytes || !validText(value) {
				return ErrInvalid
			}
		}
	}
	return nil
}

// Reduce applies the command to a private copy; callers retain the original
// document on every invalid or conflicting operation, including regrouping.
// Successful results include the validated canonical bytes for publication.
func Reduce(current Document, mutation Mutation, authority Authority) (Document, []byte, error) {
	if authority.validate() != nil || validateDocument(current) != nil || len(jsonBytes(mutation)) > MaxBatchBytes {
		return Document{}, nil, ErrInvalid
	}
	next := cloneDocument(current)
	if mutation.Operation != "replace-source" {
		provenance := authority.provenance()
		index := slices.IndexFunc(next.Sources, func(s Source) bool { return s.Provenance.SourceID == authority.SourceID })
		if index < 0 {
			if len(next.Sources) == MaxSources {
				return Document{}, nil, ErrConflict
			}
			next.Sources = append(next.Sources, Source{Provenance: provenance, Generation: 1, PolicyDigest: key(Projection)})
		} else {
			if next.Sources[index].Provenance != provenance {
				return Document{}, nil, ErrConflict
			}
		}
	}
	switch mutation.Operation {
	case "observe":
		if !authority.CanObserve || len(mutation.Observations) > MaxBatch || len(mutation.Gaps) > 16 || !onlyFields(mutation, "observe") {
			return Document{}, nil, ErrInvalid
		}
		for _, observation := range mutation.Observations {
			if err := observe(&next, observation, authority); err != nil {
				return Document{}, nil, err
			}
		}
		for _, gap := range mutation.Gaps {
			if gap.SourceID != authority.SourceID || !validGapReason(gap.Reason) || gap.Count == 0 || gap.Count > MaxOmissionCount {
				return Document{}, nil, ErrInvalid
			}
			addGap(&next, gap.SourceID, gap.Reason, gap.Count)
		}
	case "declare":
		if !onlyFields(mutation, "declare") {
			return Document{}, nil, ErrInvalid
		}
		if err := declare(&next, mutation, authority); err != nil {
			return Document{}, nil, err
		}
	case "annotate", "remove", "restore":
		if !onlyFields(mutation, mutation.Operation) {
			return Document{}, nil, ErrInvalid
		}
		if err := edit(&next, mutation, authority); err != nil {
			return Document{}, nil, err
		}
	case "replace-source":
		if !onlyFields(mutation, "replace-source") || mutation.Source == nil {
			return Document{}, nil, ErrInvalid
		}
		if err := replaceSource(&next, *mutation.Source, authority); err != nil {
			return Document{}, nil, err
		}
	default:
		return Document{}, nil, ErrInvalid
	}
	data, err := encode(next)
	if err != nil {
		return Document{}, nil, err
	}
	return next, data, nil
}

// onlyFields closes each command variant as well as the enclosing JSON type.
func onlyFields(m Mutation, op string) bool {
	clean := Mutation{Operation: m.Operation, ExpectedRevision: m.ExpectedRevision}
	switch op {
	case "observe":
		clean.Observations, clean.Gaps, clean.Artifact = m.Observations, m.Gaps, m.Artifact
	case "declare":
		clean.URL, clean.Method = m.URL, m.Method
	case "annotate":
		clean.MemberIDs, clean.Note, clean.Tags = m.MemberIDs, m.Note, m.Tags
	case "remove":
		clean.MarkerID, clean.MemberIDs = m.MarkerID, m.MemberIDs
	case "restore":
		clean.MarkerID = m.MarkerID
	case "replace-source":
		clean.Source = m.Source
	}
	return reflect.DeepEqual(clean, m)
}

func addGap(d *Document, source, reason string, count uint64) {
	for i := range d.Gaps {
		if d.Gaps[i].SourceID == source && d.Gaps[i].Reason == reason {
			d.Gaps[i].Count = saturated(d.Gaps[i].Count, count)
			return
		}
	}
	d.Gaps = append(d.Gaps, Gap{source, reason, count})
}

func markProjectionGap(d *Document, source string) {
	if !slices.ContainsFunc(d.Gaps, func(g Gap) bool { return g.SourceID == source && g.Reason == "projection" }) {
		addGap(d, source, "projection", 1)
	}
}
func saturated(a, b uint64) uint64 {
	if MaxOmissionCount-a < b {
		return MaxOmissionCount
	}
	return a + b
}

func ensureGroup(d *Document, origin, method, route string, structural ...StructuralValue) (*Group, error) {
	id := groupID(origin, method, route, structural...)
	for i := range d.Groups {
		if d.Groups[i].ID == id {
			return &d.Groups[i], nil
		}
	}
	if len(d.Groups) == MaxGroups {
		return nil, ErrConflict
	}
	d.Groups = append(d.Groups, Group{ID: id, Origin: origin, Method: method, Route: route, Structural: append([]StructuralValue{}, structural...), Members: []Member{}, Declarations: []Declaration{}})
	return &d.Groups[len(d.Groups)-1], nil
}

// resolveMemberSelection applies the same route and finite-selector policy to
// observations and manual declarations. Neither arrival order creates a second
// identity for the same concrete request.
func resolveMemberSelection(d *Document, origin, method, path, raw, source string) (string, []StructuralValue, bool, error) {
	route, ambiguous := routeFor(d.Rules, origin, method, path)
	if ambiguous {
		addGap(d, source, "ambiguous", 1)
	}
	structural, err := projectStructural(d.Rules, origin, method, route, raw)
	if errors.Is(err, errStructuralSelection) {
		addGap(d, source, "structural", 1)
		return route, nil, true, nil
	}
	return route, structural, false, err
}

// ensureMember preserves the globally unique concrete identity, including its
// existing annotations, evidence and suppression selection if placement changes.
// The containing group is already admitted; no group append invalidates target.
func ensureMember(d *Document, target *Group, path string) (*Member, error) {
	id := memberID(target.Origin, target.Method, path, target.Structural...)
	if index := slices.IndexFunc(target.Members, func(m Member) bool { return m.ID == id }); index >= 0 {
		return &target.Members[index], nil
	}
	if len(target.Members) == MaxGroupMembers {
		return nil, ErrConflict
	}
	count := 0
	for i := range d.Groups {
		group := &d.Groups[i]
		count += len(group.Members)
		if index := slices.IndexFunc(group.Members, func(m Member) bool { return m.ID == id }); index >= 0 {
			target.Members = append(target.Members, group.Members[index])
			group.Members = slices.Delete(group.Members, index, index+1)
			return &target.Members[len(target.Members)-1], nil
		}
	}
	if count == MaxMembers {
		return nil, ErrConflict
	}
	target.Members = append(target.Members, Member{ID: id, Path: path, Summaries: []Summary{}, Examples: []string{}, Evidence: []Evidence{}, Annotations: []Annotation{}})
	return &target.Members[len(target.Members)-1], nil
}

func declare(d *Document, mutation Mutation, authority Authority) error {
	if !methodPattern.MatchString(mutation.Method) {
		return ErrInvalid
	}
	origin, path, parameters, err := projectURL(mutation.URL)
	if err != nil {
		return err
	}
	if projectedLocations(mutation.URL) {
		markProjectionGap(d, authority.SourceID)
	}
	route, structural, omitted, err := resolveMemberSelection(d, origin, mutation.Method, path, mutation.URL, authority.SourceID)
	if err != nil || omitted {
		return err
	}
	target, err := ensureGroup(d, origin, mutation.Method, route, structural...)
	if err != nil {
		return err
	}
	if _, err := ensureMember(d, target, path); err != nil {
		return err
	}
	// A declaration records the concrete path, independently of a template that
	// groups its member. Projection and regrouping can recover its exact member
	// even after observations or the template source are excluded.
	concrete, err := ensureGroup(d, origin, mutation.Method, path, structural...)
	if err != nil {
		return err
	}
	declaration := Declaration{Provenance: authority.provenance(), Pointer: "manual", Parameters: parameters}
	if !slices.ContainsFunc(concrete.Declarations, func(v Declaration) bool { return reflect.DeepEqual(v, declaration) }) {
		if len(concrete.Declarations) == MaxRules {
			return ErrConflict
		}
		concrete.Declarations = append(concrete.Declarations, declaration)
	}
	return nil
}

func observe(d *Document, o Observation, a Authority) error {
	if !methodPattern.MatchString(o.Method) || !validOutcome(o.Outcome, o.Status) || !validTime(o.At) || len(o.Parameters) > MaxBatch {
		return ErrInvalid
	}
	origin, path, parameters, err := projectURL(o.URL)
	if err != nil {
		return err
	}
	if projectedLocations(o.URL) {
		markProjectionGap(d, a.SourceID)
	}
	for _, name := range o.Parameters {
		if validLabel(name) {
			parameters = append(parameters, name)
		} else {
			addGap(d, a.SourceID, "parameters", 1)
		}
	}
	slices.Sort(parameters)
	parameters = slices.Compact(parameters)
	if len(parameters) > MaxBatch {
		return ErrInvalid
	}
	route, structural, omitted, err := resolveMemberSelection(d, origin, o.Method, path, o.URL, a.SourceID)
	if err != nil || omitted {
		return err
	}
	g, err := ensureGroup(d, origin, o.Method, route, structural...)
	if err != nil {
		addGap(d, a.SourceID, "groups", 1)
		return nil
	}
	m, err := ensureMember(d, g, path)
	if err != nil {
		g.Omitted = saturated(g.Omitted, 1)
		addGap(d, a.SourceID, "members", 1)
		return nil
	}
	summary := Summary{Provenance: a.provenance(), Outcome: o.Outcome, Status: o.Status, Parameters: parameters, First: o.At, Last: o.At}
	for i := range m.Summaries {
		if summaryKey(m.Summaries[i]) == summaryKey(summary) {
			if timeBefore(o.At, m.Summaries[i].First) {
				m.Summaries[i].First = o.At
			}
			if timeBefore(m.Summaries[i].Last, o.At) {
				m.Summaries[i].Last = o.At
			}
			return nil
		}
	}
	examples := 0
	for _, member := range g.Members {
		examples += len(member.Examples)
	}
	if len(m.Summaries) == MaxSummaries {
		m.Omitted = saturated(m.Omitted, 1)
		addGap(d, a.SourceID, "summaries", 1)
		return nil
	}
	m.Summaries = append(m.Summaries, summary)
	example := o.Method + " " + origin + path
	if examples < MaxExamples && len(example) <= MaxExampleBytes && !slices.Contains(m.Examples, example) {
		m.Examples = append(m.Examples, example)
	}
	return nil
}

func edit(d *Document, m Mutation, a Authority) error {
	if m.Operation == "restore" {
		index := slices.IndexFunc(d.Markers, func(v Marker) bool { return v.ID == m.MarkerID })
		if index < 0 {
			return ErrConflict
		}
		d.Markers = slices.Delete(d.Markers, index, index+1)
		return nil
	}
	if len(m.MemberIDs) == 0 || len(m.MemberIDs) > MaxGroupMembers {
		return ErrInvalid
	}
	selection := map[string]bool{}
	for _, id := range m.MemberIDs {
		if selection[id] {
			return ErrInvalid
		}
		selection[id] = true
	}
	annotation, err := projectAnnotation(m.Note, m.Tags, a)
	if err != nil {
		return err
	}
	for i := range d.Groups {
		for j := range d.Groups[i].Members {
			member := &d.Groups[i].Members[j]
			if selection[member.ID] {
				delete(selection, member.ID)
				if m.Operation == "annotate" {
					idx := slices.IndexFunc(member.Annotations, func(v Annotation) bool { return v.Provenance == a.provenance() })
					if idx < 0 {
						member.Annotations = append(member.Annotations, annotation)
					} else {
						member.Annotations[idx] = annotation
					}
				}
			}
		}
	}
	if len(selection) > 0 {
		return ErrConflict
	}
	if m.Operation == "remove" {
		if !contracts.ValidResourceID(m.MarkerID) || len(d.Markers) == MaxMarkers || slices.ContainsFunc(d.Markers, func(v Marker) bool { return v.ID == m.MarkerID }) {
			return ErrConflict
		}
		ids := slices.Clone(m.MemberIDs)
		slices.Sort(ids)
		d.Markers = append(d.Markers, Marker{m.MarkerID, ids, a.provenance()})
	}
	return nil
}

func routeFor(rules []Rule, origin, method, path string) (string, bool) {
	matches := map[string]bool{}
	for _, rule := range rules {
		if rule.Origin != origin || rule.Method != method {
			continue
		}
		if rule.Path == path {
			return path, false
		}
		pattern, concrete := strings.Split(rule.Path, "/"), strings.Split(path, "/")
		if len(pattern) != len(concrete) {
			continue
		}
		match := true
		for i, segment := range pattern {
			if strings.HasPrefix(segment, "{") {
				if concrete[i] == "" {
					match = false
				}
			} else if segment != concrete[i] {
				match = false
			}
		}
		if match {
			matches[rule.Path] = true
		}
	}
	if len(matches) == 1 {
		for route := range matches {
			return route, false
		}
	}
	return path, len(matches) > 1
}

func replaceSource(d *Document, u SourceUpdate, a Authority) error {
	if !contracts.ValidResourceID(u.SourceID) || u.SourceID != a.SourceID || len(u.Rules) > MaxRules || observedKind(a.Kind) {
		return ErrInvalid
	}
	if u.Artifact != nil && u.Artifact.Validate() != nil {
		return ErrInvalid
	}
	index := slices.IndexFunc(d.Sources, func(s Source) bool { return s.Provenance.SourceID == u.SourceID })
	generation := uint64(1)
	if index < 0 {
		if u.Expected != nil {
			return ErrConflict
		}
		if len(d.Sources) == MaxSources {
			return ErrConflict
		}
	} else {
		if u.Expected == nil || !reflect.DeepEqual(d.Sources[index], *u.Expected) || d.Sources[index].Generation == MaxOmissionCount {
			return ErrConflict
		}
		previous := d.Sources[index].Provenance
		if previous.Kind != a.Kind || previous.Deployment != a.Deployment {
			return ErrConflict
		}
		generation = d.Sources[index].Generation + 1
	}
	u.Rules = canonicalRules(u.Rules)
	for _, rule := range u.Rules {
		if rule.SourceID != u.SourceID || validateRule(rule) != nil || !validText(rule.Path) || !validOrigin(rule.Origin) || !validText(rule.Pointer) {
			return ErrInvalid
		}
	}
	source := Source{Provenance: a.provenance(), Generation: generation, Artifact: u.Artifact, PolicyDigest: groupingPolicyDigest(u.Rules)}
	if index < 0 {
		d.Sources = append(d.Sources, source)
	} else {
		d.Sources[index] = source
	}
	d.Rules = slices.DeleteFunc(d.Rules, func(r Rule) bool { return r.SourceID == u.SourceID })
	d.Rules = append(d.Rules, u.Rules...)
	return regroup(d)
}

// regroup traverses retained members once. Unknown old membership is carried
// as an explicit gap, never reassigned from examples or resurrected.
func regroup(d *Document) error {
	suppressed := map[string]bool{}
	for _, marker := range d.Markers {
		for _, member := range marker.Members {
			suppressed[member] = true
		}
	}
	old := d.Groups
	d.Groups = []Group{}
	for _, g := range old {
		if g.Omitted > 0 {
			addGap(d, "", "regrouping", g.Omitted)
		}
		for _, m := range g.Members {
			route, ambiguous := routeFor(d.Rules, g.Origin, g.Method, m.Path)
			if ambiguous {
				addGap(d, "", "ambiguous", 1)
			}
			if suppressed[m.ID] && !sameStructuralNames(d.Rules, g.Origin, g.Method, route, g.Structural) {
				return ErrConflict
			}
			if len(g.Structural) == 0 && slices.ContainsFunc(d.Rules, func(r Rule) bool {
				return r.Origin == g.Origin && r.Method == g.Method && r.Path == route && len(r.Structural) > 0
			}) {
				addGap(d, "", "regrouping", 1)
			}
			target, err := ensureGroup(d, g.Origin, g.Method, route, g.Structural...)
			if err != nil || len(target.Members) == MaxGroupMembers {
				return ErrConflict
			}
			target.Members = append(target.Members, m)
		}
		for _, declaration := range g.Declarations {
			if declaration.Rule {
				continue
			}
			target, err := ensureGroup(d, g.Origin, g.Method, g.Route, g.Structural...)
			if err != nil {
				return err
			}
			target.Declarations = append(target.Declarations, declaration)
		}
	}
	for _, rule := range d.Rules {
		source := d.Sources[slices.IndexFunc(d.Sources, func(s Source) bool { return s.Provenance.SourceID == rule.SourceID })]
		g, err := ensureGroup(d, rule.Origin, rule.Method, rule.Path)
		if err != nil {
			return err
		}
		g.Declarations = append(g.Declarations, Declaration{Rule: true, Provenance: source.Provenance, Pointer: rule.Pointer, Parameters: rule.Parameters})
	}
	for i := range d.Groups {
		group := &d.Groups[i]
		slices.SortFunc(group.Members, func(a, b Member) int { return strings.Compare(a.ID, b.ID) })
		remaining := MaxExamples
		for j := range group.Members {
			member := &group.Members[j]
			slices.Sort(member.Examples)
			keep := min(remaining, len(member.Examples))
			if len(member.Examples) > keep {
				addGap(d, "", "examples", uint64(len(member.Examples)-keep))
			}
			member.Examples = member.Examples[:keep]
			remaining -= keep
		}
	}
	// Merging examples/contexts beyond the lossless bound is a guarded conflict;
	// accepted member annotations, markers and observations are never discarded.
	if err := validateDocument(*d); err != nil {
		return ErrConflict
	}
	return nil
}

func validGapReason(reason string) bool {
	return slices.Contains([]string{"structural", "members", "groups", "summaries", "examples", "regrouping", "incomplete", "parameters", "source-unavailable", "ambiguous", "projection", "report-truncated", "request-errors", "evidence"}, reason)
}
