package findingintake

// SQL statements for collection_publication.go. Each one reads a whole
// publication's rows of one kind, so the number of statements a publication
// runs does not grow with its entries (LIM-FND-15).

// authorizedCollectionSourcesSQL returns the sources, passed as parallel
// kind ($2) and ID ($3) arrays, that owner $1 may read: its own Runs and
// Audits. Used by authorizedCollectionSources.
var authorizedCollectionSourcesSQL = `
SELECT source.kind, source.id
  FROM unnest($2::text[], $3::text[]) AS source(kind, id)
 WHERE (source.kind = 'run' AND EXISTS (
            SELECT 1 FROM workflow_runs AS run
             WHERE run.run_id = source.id AND run.owner_id = $1))
    OR (source.kind = 'audit' AND EXISTS (
            SELECT 1 FROM audits AS audit
             WHERE audit.audit_id = source.id AND audit.owner_id = $1))`

// authorizedCollectionReceiptsSQL returns the receipts among $2 that owner $1
// may select through source kind $3 and ID $4: the receipts of that Run, or
// the receipts an Audit originated, holds or has as finding contributions.
// Used by authorizedCollectionReceipts.
var authorizedCollectionReceiptsSQL = `
SELECT receipt.receipt_id
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.owner_id = $1 AND receipt.receipt_id = ANY($2::text[]) AND (
       ($3 = 'run' AND receipt.run_id = $4)
    OR ($3 = 'audit' AND (receipt.audit_id = $4
        OR EXISTS (SELECT 1 FROM finding_proposal_audit_holds AS hold
                    WHERE hold.receipt_id = receipt.receipt_id AND hold.audit_id = $4)
        OR EXISTS (SELECT 1 FROM audit_finding_contributions AS contribution
                    WHERE contribution.receipt_id = receipt.receipt_id
                      AND contribution.audit_id = $4))))`

// collectionHoldAuditsSQL returns the Project of every Audit among $1 that
// owner $2 owns in a Project it also owns. A hold whose Audit and Project
// match a returned row is one the owner may publish. Used by
// collectionHoldAudits.
var collectionHoldAuditsSQL = `
SELECT audit.audit_id, audit.project_id
  FROM audits AS audit
  JOIN projects AS project ON project.project_id = audit.project_id
 WHERE audit.audit_id = ANY($1::text[]) AND audit.owner_id = $2
   AND project.owner_id = $2`
