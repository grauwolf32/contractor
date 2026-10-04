package evalstore

// SQL statements for report_sources.go.

// viewSourcesSQL collects the distinct provenance of the result and assessment
// records selected by members of the owner's view generation $3. Result
// sources are returned as stored; assessment sources are reduced to a system
// and ID (producer ID, 'owner-review' or 'native-checks') with null revision
// and digest. Ordered, limited to $4. Used by Store.ViewSources.
var viewSourcesSQL = `
WITH selected AS (
    SELECT m.member_id, convert_from(m.document, 'UTF8')::jsonb AS document
    FROM eval_view_members m
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND m.experiment_id = $2 AND m.generation = $3
), sources AS (
    SELECT r.kind, convert_from(r.document, 'UTF8')::jsonb -> 'source' AS source
    FROM selected m
    CROSS JOIN LATERAL (VALUES
        ('result', m.document ->> 'resultSha256'),
        ('assessment', m.document ->> 'assessmentSha256')
    ) chosen(kind, digest)
    JOIN eval_records r ON r.experiment_id = $2
        AND r.member_id = m.member_id
        AND r.kind = chosen.kind
        AND r.record_sha256 = chosen.digest
)
SELECT DISTINCT CASE WHEN kind = 'result' THEN source
    ELSE jsonb_build_object(
        'system', source ->> 'kind',
        'id', CASE source ->> 'kind'
            WHEN 'external' THEN source ->> 'producerId'
            WHEN 'human' THEN 'owner-review'
            ELSE 'native-checks'
        END,
        'revision', NULL, 'sourceSha256', NULL
    )
END AS provenance
FROM sources
ORDER BY provenance
LIMIT $4
`
