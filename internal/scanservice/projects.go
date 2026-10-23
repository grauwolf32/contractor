package scanservice

import (
	"context"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/jackc/pgx/v5"
)

func captureProjects(ctx context.Context, tx pgx.Tx, owner string, r Request) ([]scanstore.CapturedProject, error) {
	sql := selectedProjectsSQL
	args := []any{owner, r.ProjectIDs}
	if r.AllProjects {
		sql = allProjectsSQL
		args = []any{owner}
	}
	rows, err := tx.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	found := map[string]scanstore.CapturedProject{}
	result := []scanstore.CapturedProject{}
	for rows.Next() {
		var p scanstore.CapturedProject
		var lifecycle string
		if err := rows.Scan(&p.ProjectID, &lifecycle, &p.Target); err != nil {
			return nil, err
		}
		if lifecycle != "active" {
			p.Reason = "project_deleting"
		} else if p.Target == "" {
			p.Reason = "target_missing"
		}
		found[p.ProjectID] = p
		result = append(result, p)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if r.AllProjects {
		return result, nil
	}
	result = make([]scanstore.CapturedProject, 0, len(r.ProjectIDs))
	for _, id := range r.ProjectIDs {
		p, ok := found[id]
		if !ok {
			p = scanstore.CapturedProject{ProjectID: id, Reason: "project_unavailable"}
		}
		result = append(result, p)
	}
	return result, nil
}
