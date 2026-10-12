package agentskills

// Data returns a copy of a validated member's bytes.
func (m Member) Data() []byte { return append([]byte(nil), m.data...) }

// Members returns a copy of the package's validated members.
func (p *Package) Members() []Member {
	result := make([]Member, len(p.members))
	copy(result, p.members)
	return result
}

// Member returns the validated member at path.
func (p *Package) Member(path string) (Member, bool) {
	for _, member := range p.members {
		if member.Path == path {
			return member, true
		}
	}
	return Member{}, false
}
