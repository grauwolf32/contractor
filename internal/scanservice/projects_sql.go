package scanservice

// allProjectsSQL reads only the ordered prefix of the owner's project index.
// The extra row rejects oversized scopes without sorting the whole portfolio.
const allProjectsSQL = `SELECT project_id,lifecycle_state,COALESCE(http_target_url,'')
FROM projects WHERE owner_id=$1 AND kind='project'
ORDER BY project_id COLLATE "C" LIMIT 1001`

// selectedProjectsSQL performs at most 1000 primary-key point lookups, even
// with a generic prepared plan or a much larger containing owner portfolio.
const selectedProjectsSQL = `SELECT p.project_id,p.lifecycle_state,COALESCE(p.http_target_url,'')
FROM unnest($2::text[]) AS selected(project_id)
CROSS JOIN LATERAL (
 SELECT project_id,lifecycle_state,http_target_url FROM projects
 WHERE project_id=selected.project_id AND owner_id=$1 AND kind='project'
 LIMIT 1
) p`
