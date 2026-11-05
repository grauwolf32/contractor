package pgstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func scanHypothesis(row scanner, extra ...any) (auditstore.Hypothesis, error) {
	var result auditstore.Hypothesis
	var data []byte
	if err := row.Scan(append([]any{&data, &result.SourceKey, &result.Fixed, &result.Ordinal}, extra...)...); err != nil {
		return result, err
	}
	h, err := auditcontract.DecodeHypothesis(data)
	if err != nil || result.Ordinal < 1 || result.Ordinal > 10000 || !contracts.ValidName(result.SourceKey) || result.Fixed && h.Origin.Kind != "checklist" {
		return auditstore.Hypothesis{}, errStoredTask
	}
	result.Document = h
	return result, nil
}

func (s *QueryStore) GetHypothesis(ctx context.Context, ownerID, auditID, id string, revision uint64) (auditstore.Hypothesis, error) {
	if !contracts.ValidResourceID(id) || revision == 0 || revision > contracts.MaxExactInteger {
		return auditstore.Hypothesis{}, auditstore.ErrInvalid
	}
	audit, err := s.get(ctx, ownerID, auditID)
	if err != nil {
		return auditstore.Hypothesis{}, err
	}
	pinned, err := decodeAuditProgram(audit)
	if err != nil {
		return auditstore.Hypothesis{}, err
	}
	result, err := scanHypothesis(s.db.QueryRow(ctx, exactHypothesisSQL, auditID, id, revision))
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	if err == nil && (result.Document.ID != id || result.Document.Revision != revision || result.Document.Origin.AuditID != auditID || result.Ordinal > pinned.Program.Limits.MaxHypotheses) {
		return auditstore.Hypothesis{}, errStoredTask
	}
	return result, err
}

func (s *QueryStore) ListHypotheses(ctx context.Context, p auditstore.ListHypothesesParams) (auditstore.HypothesisPage, error) {
	if !pagelimit.ValidPage(p.Limit) || p.AfterOrdinal < 0 {
		return auditstore.HypothesisPage{}, auditstore.ErrInvalid
	}
	audit, err := s.get(ctx, p.OwnerID, p.AuditID)
	if err != nil {
		return auditstore.HypothesisPage{}, err
	}
	pinned, err := decodeAuditProgram(audit)
	if err != nil {
		return auditstore.HypothesisPage{}, err
	}
	rows, err := s.db.Query(ctx, hypothesisPageSQL, p.AuditID, p.AfterOrdinal, pagelimit.FetchRows(p.Limit))
	if err != nil {
		return auditstore.HypothesisPage{}, err
	}
	defer rows.Close()
	page := auditstore.HypothesisPage{PendingFixedCount: max(0, len(pinned.Program.Hypotheses)-audit.InitialHypothesisOffset), FixedDenominator: len(pinned.Program.Hypotheses), Items: []auditstore.Hypothesis{}}
	page.CheckingWorkflows = []string{}
	for alias, binding := range pinned.Program.Workflows {
		checking := binding.Outputs["result"] != ""
		hasContext := false
		for _, input := range binding.Inputs {
			hasContext = hasContext || input.Source == "task-context"
			if input.Source == "task-input" || input.Source == "prior-results" {
				checking = false
			}
		}
		if checking && hasContext {
			page.CheckingWorkflows = append(page.CheckingWorkflows, alias)
		}
	}
	sort.Strings(page.CheckingWorkflows)
	var count int
	for rows.Next() {
		h, err := scanHypothesis(rows, &count)
		if err != nil {
			return page, err
		}
		if h.Document.Origin.AuditID != p.AuditID || h.Ordinal > pinned.Program.Limits.MaxHypotheses {
			return page, errStoredTask
		}
		page.Items = append(page.Items, h)
	}
	if err := rows.Err(); err != nil {
		return page, err
	}
	if count > len(page.Items) || len(page.Items) > p.Limit {
		if len(page.Items) > p.Limit {
			page.Items = page.Items[:p.Limit]
		}
		if len(page.Items) == 0 {
			return page, errStoredTask
		}
		last := page.Items[len(page.Items)-1].Ordinal
		page.NextOrdinal = &last
	}
	return page, nil
}

// persistHypotheses runs under the locked Audit row, charging cumulative
// durable usage. It is shared by manual writes and initial Task acceptance.
func persistHypotheses(ctx context.Context, tx pgx.Tx, audit auditstore.Audit, pinned auditprofile.PinnedProgram, writes []auditstore.HypothesisWrite, initial bool) error {
	if len(writes) > auditcontract.MaxMutationRecords {
		return auditstore.ErrInvalid
	}
	var count, revisions int
	var used int64
	if err := tx.QueryRow(ctx, hypothesisUsageSQL, audit.AuditID).Scan(&count, &revisions, &used); err != nil {
		return err
	}
	seen := map[string]bool{}
	sliceBytes := 0
	for _, write := range writes {
		data, err := strictjson.Canonical(write.Document)
		sliceBytes += len(data)
		if err != nil || sliceBytes > auditcontract.MaxMutationBytes || seen[write.Document.ID] || !contracts.ValidName(write.SourceKey) {
			return auditstore.ErrInvalid
		}
		seen[write.Document.ID] = true
		h, err := auditcontract.DecodeHypothesis(data)
		if err != nil || h.Origin.AuditID != audit.AuditID {
			return auditstore.ErrInvalid
		}
		var current uint64
		var source string
		var fixed bool
		err = tx.QueryRow(ctx, `SELECT current_revision,source_key,fixed FROM audit_hypotheses WHERE hypothesis_id=$1 AND audit_id=$2`, h.ID, audit.AuditID).Scan(&current, &source, &fixed)
		if errors.Is(err, pgx.ErrNoRows) {
			if h.Revision != 1 || !initial && (h.Origin.Kind != "manual" || h.Origin.Source != nil || write.Fixed) {
				return auditstore.ErrInvalid
			}
			count++
			if _, err := tx.Exec(ctx, `INSERT INTO audit_hypotheses(hypothesis_id,audit_id,source_key,fixed,ordinal,current_revision,current_bytes) VALUES($1,$2,$3,$4,$5,1,$6)`, h.ID, audit.AuditID, write.SourceKey, write.Fixed, count, len(data)); err != nil {
				return writeError("create hypothesis", err)
			}
		} else if err != nil {
			return err
		} else {
			if source != write.SourceKey || fixed != write.Fixed {
				return auditstore.ErrConflict
			}
			readRevision := current
			if initial {
				readRevision = 1
			}
			prior, err := scanHypothesis(tx.QueryRow(ctx, exactHypothesisSQL, audit.AuditID, h.ID, readRevision))
			if err != nil {
				return err
			}
			if initial {
				old, _ := strictjson.Canonical(prior.Document)
				if !bytes.Equal(old, data) {
					return auditstore.ErrConflict
				}
				continue
			}
			a, _ := strictjson.Canonical(prior.Document.Origin)
			b, _ := strictjson.Canonical(h.Origin)
			if h.Revision != current+1 || !bytes.Equal(a, b) {
				return auditstore.ErrPrecondition
			}
			if _, err := tx.Exec(ctx, `UPDATE audit_hypotheses SET current_revision=$2,current_bytes=$3 WHERE hypothesis_id=$1`, h.ID, h.Revision, len(data)); err != nil {
				return err
			}
		}
		revisions++
		used += int64(len(data))
		limits := pinned.Program.Limits
		if count > limits.MaxHypotheses || revisions > limits.MaxHypothesisRevisions || used > limits.MaxHypothesisBytes {
			return auditstore.ErrInvalid
		}
		if _, err := tx.Exec(ctx, `INSERT INTO audit_hypothesis_revisions(hypothesis_id,audit_id,revision,document) VALUES($1,$2,$3,$4::json)`, h.ID, audit.AuditID, h.Revision, data); err != nil {
			return err
		}
	}
	_, err := tx.Exec(ctx, updateHypothesisUsageSQL, audit.AuditID, count, revisions, used)
	return err
}

func (s *OwnerStore) WriteHypotheses(ctx context.Context, p auditstore.WriteHypothesesParams) (auditstore.HypothesisAcceptance, bool, error) {
	if p.OwnerID == "" || !contracts.ValidResourceID(p.AuditID) || (p.Claim == nil && p.ExpectedRevision == 0 || p.Claim != nil && (validateClaimIdentity(*p.Claim) != nil || p.Claim.AuditID != p.AuditID || p.InitialProgress == nil)) || validateIdempotency(p.IdempotencyKey, p.RequestDigest) != nil || len(p.Writes) == 0 || len(p.Writes) > auditcontract.MaxMutationRecords {
		return auditstore.HypothesisAcceptance{}, false, auditstore.ErrInvalid
	}
	var result auditstore.HypothesisAcceptance
	var replayed bool
	err := postgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		receipt, replay, err := postgres.KeyedMutation(ctx, tx, auditReceiptKey(p.OwnerID, auditstore.MutationWriteHypotheses, p.IdempotencyKey), p.RequestDigest, func() (postgres.Receipt, error) {
			var projectID string
			if err := tx.QueryRow(ctx, `SELECT project_id FROM audits WHERE audit_id=$1 AND owner_id=$2`, p.AuditID, p.OwnerID).Scan(&projectID); err != nil {
				if errors.Is(err, pgx.ErrNoRows) {
					err = auditstore.ErrNotFound
				}
				return postgres.Receipt{}, err
			}
			if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, p.OwnerID, projectID); err != nil {
				return postgres.Receipt{}, err
			}
			if p.Claim != nil {
				if err := auditClaims.Lock(ctx, tx, p.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
					return postgres.Receipt{}, auditstore.ErrClaimLost
				}
			}
			audit, err := scanAudit(tx.QueryRow(ctx, `SELECT `+prefixedAuditColumns("audit")+` FROM audits audit WHERE audit_id=$1 AND owner_id=$2 FOR UPDATE`, p.AuditID, p.OwnerID))
			if err != nil {
				return postgres.Receipt{}, err
			}
			if p.Claim == nil && audit.Revision != p.ExpectedRevision || audit.State != auditstore.AuditActive && audit.State != auditstore.AuditPaused {
				return postgres.Receipt{}, auditstore.ErrPrecondition
			}
			pinned, err := decodeAuditProgram(audit)
			if err != nil {
				return postgres.Receipt{}, err
			}
			if p.InitialProgress != nil {
				keys := auditdomain.InitialHypothesisKeys(pinned.Program)
				progress := p.InitialProgress
				if p.Claim == nil || progress.From != audit.InitialHypothesisOffset || progress.To-progress.From != len(p.Writes) || progress.To > len(keys) || progress.From < 0 {
					return postgres.Receipt{}, auditstore.ErrPrecondition
				}
				var fields map[string]json.RawMessage
				var plan *auditcontract.InitialAssertions
				if strictjson.Decode(audit.BaselineSnapshot, &fields) != nil {
					return postgres.Receipt{}, errStoredTask
				}
				if raw := fields["initial_assertions"]; raw != nil {
					if strictjson.Decode(raw, &plan) != nil {
						return postgres.Receipt{}, errStoredTask
					}
				}
				for i, w := range p.Writes {
					key := keys[progress.From+i]
					h := auditdomain.InitialHypothesis(audit.AuditID, key, pinned.Program.Hypotheses[key])
					if plan != nil {
						if _, found := plan.Hypotheses[key]; found {
							h.Origin.Source = &plan.Source
						}
					}
					a, _ := strictjson.Canonical(h)
					b, _ := strictjson.Canonical(w.Document)
					if w.SourceKey != key || !w.Fixed || string(a) != string(b) {
						return postgres.Receipt{}, auditstore.ErrInvalid
					}
				}
				if _, err := tx.Exec(ctx, `UPDATE audits SET initial_hypothesis_offset=$2 WHERE audit_id=$1`, p.AuditID, progress.To); err != nil {
					return postgres.Receipt{}, err
				}
			}
			if err := persistHypothesisSources(ctx, tx, audit, p.Writes, s.storage); err != nil {
				return postgres.Receipt{}, err
			}
			if err := persistHypotheses(ctx, tx, audit, pinned, p.Writes, p.InitialProgress != nil); err != nil {
				return postgres.Receipt{}, err
			}
			result = auditstore.HypothesisAcceptance{Hypotheses: []auditcontract.HypothesisRef{}}
			for _, w := range p.Writes {
				result.Hypotheses = append(result.Hypotheses, auditcontract.HypothesisRef{ID: w.Document.ID, Revision: w.Document.Revision})
			}
			var revision, sequence int64
			if err := tx.QueryRow(ctx, advanceHypothesesSQL, p.AuditID, len(p.Writes)).Scan(&revision, &sequence); err != nil {
				return postgres.Receipt{}, err
			}
			if p.Claim != nil {
				live, err := auditClaims.Live(ctx, tx, p.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch))
				if err != nil {
					return postgres.Receipt{}, err
				}
				if !live {
					return postgres.Receipt{}, auditstore.ErrClaimLost
				}
			}
			result.AuditRevision = uint64(revision)
			data, err := strictjson.Canonical(result)
			return postgres.Receipt{Operation: string(auditstore.MutationWriteHypotheses), ResourceID: p.AuditID, OwnerID: p.OwnerID, AuditID: p.AuditID, Response: data}, err
		})
		if err != nil {
			return auditReceiptError(err, auditstore.MutationWriteHypotheses)
		}
		if receipt.AuditID != p.AuditID {
			return contracts.ErrIdempotencyConflict
		}
		if err := decodeHypothesisAcceptance(receipt.Response, &result); err != nil {
			return errStoredTask
		}
		replayed = replay
		return nil
	})
	return result, replayed, err
}

func (s *QueryStore) CurrentHypothesis(ctx context.Context, ownerID, auditID, id string) (auditstore.Hypothesis, error) {
	if _, err := s.get(ctx, ownerID, auditID); err != nil {
		return auditstore.Hypothesis{}, err
	}
	var revision uint64
	if err := s.db.QueryRow(ctx, `SELECT current_revision FROM audit_hypotheses WHERE audit_id=$1 AND hypothesis_id=$2`, auditID, id).Scan(&revision); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			err = auditstore.ErrNotFound
		}
		return auditstore.Hypothesis{}, err
	}
	return s.GetHypothesis(ctx, ownerID, auditID, id, revision)
}

func (s *OwnerStore) LookupHypothesisAcceptance(ctx context.Context, ownerID, auditID, key, digest string) (auditstore.HypothesisAcceptance, bool, error) {
	if ownerID == "" || !contracts.ValidResourceID(auditID) || validateIdempotency(key, digest) != nil {
		return auditstore.HypothesisAcceptance{}, false, auditstore.ErrInvalid
	}
	receipt, found, err := postgres.FindReceipt(ctx, s.db, auditReceiptKey(ownerID, auditstore.MutationWriteHypotheses, key), digest)
	if err != nil || !found {
		return auditstore.HypothesisAcceptance{}, false, auditReceiptError(err, auditstore.MutationWriteHypotheses)
	}
	if receipt.AuditID != auditID {
		return auditstore.HypothesisAcceptance{}, false, contracts.ErrIdempotencyConflict
	}
	var result auditstore.HypothesisAcceptance
	if err := decodeHypothesisAcceptance(receipt.Response, &result); err != nil {
		return result, false, errStoredTask
	}
	return result, true, nil
}

func decodeHypothesisAcceptance(data []byte, result *auditstore.HypothesisAcceptance) error {
	if strictjson.Decode(data, result) != nil || len(result.Hypotheses) == 0 || len(result.Hypotheses) > auditcontract.MaxMutationRecords || result.AuditRevision == 0 || result.AuditRevision > contracts.MaxExactInteger {
		return errStoredTask
	}
	seen := map[string]bool{}
	for _, ref := range result.Hypotheses {
		if !contracts.ValidResourceID(ref.ID) || ref.Revision == 0 || ref.Revision > contracts.MaxExactInteger || seen[ref.ID] {
			return errStoredTask
		}
		seen[ref.ID] = true
	}
	return nil
}
