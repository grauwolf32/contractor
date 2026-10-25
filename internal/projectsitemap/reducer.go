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
	"github.com/grauwolf32/contractor/internal/strictjson"
)

var (
	ErrInvalid  = apierror.New(apierror.InvalidRequest, "invalid sitemap document or mutation")
	ErrConflict = apierror.New(apierror.Conflict, "sitemap edit or source precondition conflict")
	ErrUnsafe   = apierror.New(apierror.InvalidRequest, "sitemap input cannot form a safe concrete identity")
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
		if _, found := sources[rule.SourceID]; !found {
			return ErrInvalid
		}
		if validateRule(rule) != nil {
			return ErrInvalid
		}
	}
	groups, members := map[string]bool{}, map[string]bool{}
	for _, g := range d.Groups {
		if g.ID != groupID(g.Origin, g.Method, g.Route, g.Structural...) || groups[g.ID] || !methodPattern.MatchString(g.Method) || !validStructural(g.Structural) ||
			len(g.Members) > MaxGroupMembers || len(g.Declarations) > MaxRules || g.Omitted > MaxOmissionCount || !validRoute(g.Origin, g.Route) {
			return ErrInvalid
		}
		groups[g.ID] = true
		contexts, examples := map[string]bool{}, 0
		for _, m := range g.Members {
			if m.ID != memberID(g.Origin, g.Method, m.Path, g.Structural...) || members[m.ID] ||
				len(m.Summaries) > MaxSummaries || len(m.Evidence) > MaxProvenance || len(m.Annotations) > MaxSources ||
				len(m.Path) > MaxMemberKeyBytes || m.Omitted > MaxOmissionCount || !validRoute(g.Origin, m.Path) {
				return ErrInvalid
			}
			members[m.ID] = true
			seenSummaries := map[string]bool{}
			for _, s := range m.Summaries {
				if !observedKind(s.Provenance.Kind) || !admittedProvenance(s.Provenance, sources) || !hashPattern.MatchString(s.Context) || seenSummaries[summaryKey(s)] || !validOutcome(s.Outcome, s.Status) ||
					!validTime(s.First) || !validTime(s.Last) || timeBefore(s.Last, s.First) || !validNames(s.Parameters) {
					return ErrInvalid
				}
				contexts[s.Context] = true
				seenSummaries[summaryKey(s)] = true
			}
			examples += len(m.Examples)
			for _, example := range m.Examples {
				if len(example) > MaxExampleBytes || !safeText(example, nil) || strings.Contains(example, "?") {
					return ErrInvalid
				}
			}
			for _, evidence := range m.Evidence {
				if !contracts.ValidResourceID(evidence.RunID) || evidence.Artifact.Validate() != nil {
					return ErrInvalid
				}
			}
			for _, annotation := range m.Annotations {
				if !admittedProvenance(annotation.Provenance, sources) || len(annotation.Note) > MaxNoteBytes ||
					!safeText(annotation.Note, nil) || secretName.MatchString(annotation.Note) || len(annotation.Tags) > MaxTags || !validNames(annotation.Tags) {
					return ErrInvalid
				}
			}
		}
		if len(contexts) > MaxContexts || examples > MaxExamples {
			return ErrInvalid
		}
		for _, declaration := range g.Declarations {
			if !admittedProvenance(declaration.Provenance, sources) || !validNames(declaration.Parameters) || len(declaration.Pointer) > MaxExampleBytes ||
				!safeText(declaration.Pointer, nil) {
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
			!slices.Contains([]string{"unsafe", "members", "groups", "summaries", "contexts", "examples", "metadata", "regrouping", "incomplete", "parameters", "source-unavailable", "ambiguous", "projection"}, gap.Reason) {
			return ErrInvalid
		}
		gapIDs[id] = true
	}
	return nil
}

func validProvenance(p Provenance) bool {
	return (Authority{SourceID: p.SourceID, Kind: p.Kind, Mode: p.Mode, RunID: p.RunID, Deployment: p.Deployment, CredentialRevision: p.CredentialRevision, SessionGeneration: p.SessionGeneration}).validate() == nil
}
func admittedProvenance(p Provenance, sources map[string]Source) bool {
	source, found := sources[p.SourceID]
	return found && validProvenance(p) && source.Provenance.Kind == p.Kind && source.Provenance.Mode == p.Mode && source.Provenance.Deployment == p.Deployment
}
func validNames(names []string) bool {
	if len(names) > MaxBatch {
		return false
	}
	for _, name := range names {
		if !safeName(name, nil) {
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
	if !strings.HasPrefix(path, "/") || strings.ContainsAny(path, "?#") || len(origin)+len(path) > MaxMemberKeyBytes || !safeText(path, nil) {
		return false
	}
	o, _, _, err := projectURL(origin+"/", nil)
	return err == nil && o == origin
}
func validateRule(rule Rule) error {
	if !contracts.ValidResourceID(rule.SourceID) || !methodPattern.MatchString(rule.Method) || !validRoute(rule.Origin, rule.Path) ||
		!validNames(rule.Parameters) || len(rule.Pointer) > MaxExampleBytes || !safeText(rule.Pointer, nil) {
		return ErrInvalid
	}
	for _, segment := range strings.Split(rule.Path, "/") {
		if strings.ContainsAny(segment, "{}") && !(strings.HasPrefix(segment, "{") && strings.HasSuffix(segment, "}") && safeName(segment[1:len(segment)-1], nil)) {
			return ErrInvalid
		}
	}
	if len(rule.Structural) > MaxTags {
		return ErrInvalid
	}
	names := map[string]bool{}
	for _, selector := range rule.Structural {
		if names[selector.Name] || !safeName(selector.Name, nil) || len(selector.Allowed) == 0 || len(selector.Allowed) > MaxTags {
			return ErrInvalid
		}
		names[selector.Name] = true
		for _, value := range selector.Allowed {
			if len(value) > MaxTagBytes || !safeText(value, nil) || secretName.MatchString(value) {
				return ErrInvalid
			}
		}
	}
	return nil
}

// Reduce applies the command to a private copy; callers retain the original
// document on every invalid or conflicting operation, including regrouping.
func Reduce(current Document, mutation Mutation, authority Authority) (Document, error) {
	if authority.validate() != nil || validateDocument(current) != nil || len(jsonBytes(mutation)) > MaxBatchBytes {
		return Document{}, ErrInvalid
	}
	var next Document
	if err := strictjson.Decode(jsonBytes(current), &next); err != nil {
		return Document{}, ErrInvalid
	}
	if mutation.Operation != "replace-source" {
		provenance := authority.provenance()
		if authority.CanObserve {
			provenance.RunID, provenance.CredentialRevision, provenance.SessionGeneration = "", "", ""
		}
		index := slices.IndexFunc(next.Sources, func(s Source) bool { return s.Provenance.SourceID == authority.SourceID })
		if index < 0 {
			if len(next.Sources) == MaxSources {
				return Document{}, ErrConflict
			}
			next.Sources = append(next.Sources, Source{Provenance: provenance, Generation: 1, PolicyDigest: key(Projection)})
		} else if next.Sources[index].Provenance != provenance {
			return Document{}, ErrConflict
		}
	}
	switch mutation.Operation {
	case "observe":
		if !authority.CanObserve || len(mutation.Observations) == 0 || len(mutation.Observations) > MaxBatch || !onlyFields(mutation, "observe") {
			return Document{}, ErrInvalid
		}
		for _, observation := range mutation.Observations {
			if err := observe(&next, observation, authority); err != nil {
				return Document{}, err
			}
		}
	case "declare":
		if !onlyFields(mutation, "declare") {
			return Document{}, ErrInvalid
		}
		origin, path, _, err := projectURL(mutation.URL, authority.Secrets)
		if err != nil {
			return Document{}, err
		}
		if projectedLocations(mutation.URL) {
			markProjectionGap(&next, authority.SourceID)
		}
		if !methodPattern.MatchString(mutation.Method) {
			return Document{}, ErrInvalid
		}
		g, err := ensureGroup(&next, origin, mutation.Method, path)
		if err != nil {
			return Document{}, err
		}
		declaration := Declaration{Provenance: authority.provenance(), Pointer: "manual", Parameters: []string{}}
		if !slices.ContainsFunc(g.Declarations, func(v Declaration) bool { return reflect.DeepEqual(v, declaration) }) {
			g.Declarations = append(g.Declarations, declaration)
		}
		id := memberID(origin, mutation.Method, path)
		if !slices.ContainsFunc(g.Members, func(v Member) bool { return v.ID == id }) {
			if len(g.Members) == MaxGroupMembers {
				return Document{}, ErrConflict
			}
			g.Members = append(g.Members, Member{ID: id, Path: path, Summaries: []Summary{}, Examples: []string{}, Evidence: []Evidence{}, Annotations: []Annotation{}})
		}
	case "annotate", "remove", "restore":
		if !onlyFields(mutation, mutation.Operation) {
			return Document{}, ErrInvalid
		}
		if err := edit(&next, mutation, authority); err != nil {
			return Document{}, err
		}
	case "replace-source":
		if !onlyFields(mutation, "replace-source") || mutation.Source == nil {
			return Document{}, ErrInvalid
		}
		if err := replaceSource(&next, *mutation.Source, authority); err != nil {
			return Document{}, err
		}
	default:
		return Document{}, ErrInvalid
	}
	canonicalize(&next)
	if _, err := encode(next); err != nil {
		return Document{}, err
	}
	return next, nil
}

// onlyFields closes each command variant as well as the enclosing JSON type.
func onlyFields(m Mutation, op string) bool {
	clean := Mutation{Operation: m.Operation, ExpectedRevision: m.ExpectedRevision}
	switch op {
	case "observe":
		clean.Observations = m.Observations
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

func observe(d *Document, o Observation, a Authority) error {
	if !methodPattern.MatchString(o.Method) || !validOutcome(o.Outcome, o.Status) || !validTime(o.At) || len(o.Parameters) > MaxBatch {
		return ErrInvalid
	}
	origin, path, parameters, err := projectURL(o.URL, a.Secrets)
	if errors.Is(err, ErrUnsafe) {
		addGap(d, a.SourceID, "unsafe", 1)
		return nil
	}
	if err != nil {
		return err
	}
	if projectedLocations(o.URL) {
		markProjectionGap(d, a.SourceID)
	}
	for _, name := range o.Parameters {
		if safeName(name, a.Secrets) {
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
	route, ambiguous := routeFor(d.Rules, origin, o.Method, path)
	if ambiguous {
		addGap(d, a.SourceID, "ambiguous", 1)
	}
	structural, err := projectStructural(d.Rules, origin, o.Method, route, o.URL, a.Secrets)
	if errors.Is(err, ErrUnsafe) {
		addGap(d, a.SourceID, "unsafe", 1)
		return nil
	}
	if err != nil {
		return err
	}
	g, err := ensureGroup(d, origin, o.Method, route, structural...)
	if err != nil {
		addGap(d, a.SourceID, "groups", 1)
		return nil
	}
	id := memberID(origin, o.Method, path, structural...)
	memberIndex := slices.IndexFunc(g.Members, func(m Member) bool { return m.ID == id })
	if memberIndex < 0 {
		count := 0
		for _, group := range d.Groups {
			count += len(group.Members)
		}
		if len(g.Members) == MaxGroupMembers || count == MaxMembers {
			g.Omitted = saturated(g.Omitted, 1)
			addGap(d, a.SourceID, "members", 1)
			return nil
		}
		g.Members = append(g.Members, Member{ID: id, Path: path, Summaries: []Summary{}, Examples: []string{}, Evidence: []Evidence{}, Annotations: []Annotation{}})
		memberIndex = len(g.Members) - 1
	}
	m := &g.Members[memberIndex]
	summary := Summary{Provenance: a.provenance(), Context: a.contextKey(), Outcome: o.Outcome, Status: o.Status, Parameters: parameters, First: o.At, Last: o.At}
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
	contexts := map[string]bool{}
	examples := 0
	for _, member := range g.Members {
		examples += len(member.Examples)
		for _, s := range member.Summaries {
			contexts[s.Context] = true
		}
	}
	if !contexts[summary.Context] && len(contexts) == MaxContexts {
		m.Omitted = saturated(m.Omitted, 1)
		addGap(d, a.SourceID, "contexts", 1)
		return nil
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
	annotation, omitted, err := projectAnnotation(m.Note, m.Tags, a)
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
	if omitted {
		addGap(d, a.SourceID, "metadata", 1)
	}
	if m.Operation == "remove" {
		if !contracts.ValidResourceID(m.MarkerID) || !safeText(m.MarkerID, a.Secrets) || secretName.MatchString(m.MarkerID) || len(d.Markers) == MaxMarkers || slices.ContainsFunc(d.Markers, func(v Marker) bool { return v.ID == m.MarkerID }) {
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
	if !contracts.ValidResourceID(u.SourceID) || u.SourceID != a.SourceID || len(u.Rules) > MaxRules || a.Kind == "http" {
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
		if previous.Kind != a.Kind || previous.Mode != a.Mode || previous.Deployment != a.Deployment {
			return ErrConflict
		}
		generation = d.Sources[index].Generation + 1
	}
	u.Rules = canonicalRules(u.Rules)
	for _, rule := range u.Rules {
		if rule.SourceID != u.SourceID || validateRule(rule) != nil || !safeText(rule.Path, a.Secrets) || !safeText(rule.Pointer, a.Secrets) {
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
			route, _ := routeFor(d.Rules, g.Origin, g.Method, m.Path)
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
