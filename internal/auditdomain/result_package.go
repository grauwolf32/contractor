package auditdomain

const (
	CheckResultsMemberID = "check-results"
	EvidenceMemberID     = "evidence"
	JSONMediaType        = "application/json"
)

// CheckResultPackage is the fully validated, bounded result envelope. Content
// evidence remains addressable by member ID; callers never receive zip handles
// or paths that need to be opened again.
type CheckResultPackage struct {
	Package  *Package
	Results  CheckResultSet
	Evidence EvidenceEnvelope
}

// DecodeCheckResultPackage applies the fixed MVP member convention and all
// package-local referential checks. Correlation to an execution manifest and
// profile evidence contract remains the trusted importer's responsibility.
func DecodeCheckResultPackage(payload []byte) (CheckResultPackage, error) {
	pkg, err := ValidatePackage(payload)
	if err != nil {
		return CheckResultPackage{}, err
	}
	if pkg.Manifest.Kind != PackageKindCheckResults || pkg.Manifest.EntryPoint != "" {
		return CheckResultPackage{}, invalid(CodePackageInvalid, "manifest.kind")
	}
	resultMember, ok := pkg.MemberByID(CheckResultsMemberID)
	if !ok || resultMember.metadata.Path != "check-results.json" || resultMember.metadata.MediaType != JSONMediaType {
		return CheckResultPackage{}, invalid(CodePackageInvalid, "check-results")
	}
	results, err := DecodeCheckResultSet(resultMember.data)
	if err != nil {
		return CheckResultPackage{}, err
	}

	evidence := EvidenceEnvelope{Schema: EvidenceSchema, Evidence: []Evidence{}}
	if member, present := pkg.MemberByID(EvidenceMemberID); present {
		if member.metadata.Path != "evidence.json" || member.metadata.MediaType != JSONMediaType {
			return CheckResultPackage{}, invalid(CodePackageInvalid, "evidence")
		}
		evidence, err = DecodeEvidence(member.data)
		if err != nil {
			return CheckResultPackage{}, err
		}
	}

	referencedEvidence := make(map[string]struct{})
	for _, result := range results.Results {
		for _, id := range result.EvidenceIDs {
			referencedEvidence[id] = struct{}{}
		}
	}
	evidenceByID := make(map[string]Evidence, len(evidence.Evidence))
	contentMembers := make(map[string]struct{})
	for _, item := range evidence.Evidence {
		if _, referenced := referencedEvidence[item.ID]; !referenced {
			return CheckResultPackage{}, invalid(CodeResultSetInvalid, "evidence.unreferenced")
		}
		evidenceByID[item.ID] = item
		if item.ContentMemberID == "" {
			continue
		}
		if item.ContentMemberID == CheckResultsMemberID || item.ContentMemberID == EvidenceMemberID {
			return CheckResultPackage{}, invalid(CodeReferenceInvalid, "evidence.content_member_id")
		}
		if _, duplicate := contentMembers[item.ContentMemberID]; duplicate {
			return CheckResultPackage{}, invalid(CodeReferenceInvalid, "evidence.content_member_id")
		}
		if _, exists := pkg.MemberByID(item.ContentMemberID); !exists {
			return CheckResultPackage{}, invalid(CodeReferenceInvalid, "evidence.content_member_id")
		}
		contentMembers[item.ContentMemberID] = struct{}{}
	}
	for id := range referencedEvidence {
		if _, exists := evidenceByID[id]; !exists {
			return CheckResultPackage{}, invalid(CodeResultSetInvalid, "results.evidence_ids")
		}
	}
	for _, member := range pkg.Members() {
		id := member.metadata.ID
		if id == CheckResultsMemberID || id == EvidenceMemberID {
			continue
		}
		if _, referenced := contentMembers[id]; !referenced {
			return CheckResultPackage{}, invalid(CodeResultSetInvalid, "members.unreferenced")
		}
	}
	return CheckResultPackage{Package: pkg, Results: results, Evidence: evidence}, nil
}
