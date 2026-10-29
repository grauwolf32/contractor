package findingexport

import "sort"

// ResultGroups projects only the selected findings. It retains all independent
// reviews in Snapshot.Results; a group never borrows another member's rating.
// Manual target IDs outside the selection connect selected siblings without
// implicitly selecting or reading that target.
func (s Snapshot) ResultGroups() ([]ResultGroup, error) {
	if err := validateSnapshot(s); err != nil {
		return nil, err
	}
	if s.Request.IncludeDuplicates {
		groups := make([]ResultGroup, len(s.Results))
		for i, r := range s.Results {
			groups[i] = ResultGroup{Result: r, FindingIDs: []string{r.FindingID}}
		}
		return groups, nil
	}
	parent := map[string]string{}
	var root func(string) string
	root = func(id string) string {
		p, ok := parent[id]
		if !ok {
			parent[id] = id
			return id
		}
		if p != id {
			parent[id] = root(p)
		}
		return parent[id]
	}
	join := func(a, b string) { parent[root(a)] = root(b) }
	for _, r := range s.Results {
		id := "finding:" + r.FindingID
		root(id)
		if r.PrimaryKey != nil {
			join(id, "key:"+r.PrimaryKey.GroupKey)
		}
		if r.Review.State == "duplicate" {
			join(id, "finding:"+r.Review.DuplicateTargetID)
		}
	}
	members := map[string][]Result{}
	for _, r := range s.Results {
		key := root("finding:" + r.FindingID)
		members[key] = append(members[key], r)
	}
	groups := make([]ResultGroup, 0, len(members))
	for _, rows := range members {
		// A selected non-duplicate includes an explicitly selected manual
		// target; fallback to the first ID when all selected members are
		// duplicates. Never combine analyst ratings.
		sort.Slice(rows, func(i, j int) bool {
			a, b := rows[i], rows[j]
			if (a.Review.State == "duplicate") != (b.Review.State == "duplicate") {
				return a.Review.State != "duplicate"
			}
			return a.FindingID < b.FindingID
		})
		group := ResultGroup{Result: rows[0], FindingIDs: make([]string, len(rows))}
		for i, r := range rows {
			group.FindingIDs[i] = r.FindingID
		}
		sort.Strings(group.FindingIDs)
		groups = append(groups, group)
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i].Result.FindingID < groups[j].Result.FindingID })
	return groups, nil
}
