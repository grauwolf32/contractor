package evalstore

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type DatasetRevision struct {
	Metadata evaldomain.Dataset
	Document evaldomain.Frozen `json:"-"`
}

func (s *Store) PutDataset(ctx context.Context, scope Scope, revision string, document evaldomain.Frozen, id evaldomain.MutationIdentity) (Receipt, error) {
	if document.Kind() != "DatasetInput" || !resourceID.MatchString(revision) {
		return Receipt{}, evaldomain.Failure("eval_invalid")
	}
	var input evaldomain.DatasetInput
	if err := evaldomain.DecodeInto("DatasetInput", document.Bytes(), &input); err != nil {
		return Receipt{}, err
	}
	return s.mutate(ctx, scope, "datasets", "dataset-create", id, func() (Reference, error) {
		metadata, err := evaldomain.DatasetProjection(input, scope.ProjectID, revision)
		if err != nil {
			return Reference{}, err
		}
		_, err = s.db.Exec(ctx, `INSERT INTO eval_dataset_revisions(owner_id,project_id,dataset_id,revision,document,metadata) VALUES($1,$2,$3,$4,$5,$6)`, scope.OwnerID, scope.ProjectID, input.DatasetID, revision, document.Bytes(), bytesOf(metadata))
		return Reference{ID: input.DatasetID, State: revision}, normalize(err)
	})
}
func (s *Store) Dataset(ctx context.Context, scope Scope, id, revision string) (DatasetRevision, error) {
	var out DatasetRevision
	var document, metadata []byte
	err := s.db.QueryRow(ctx, `SELECT document,metadata FROM eval_dataset_revisions WHERE owner_id=$1 AND project_id=$2 AND dataset_id=$3 AND revision=$4`, scope.OwnerID, scope.ProjectID, id, revision).Scan(&document, &metadata)
	if err != nil {
		return out, normalize(err)
	}
	if err = json.Unmarshal(metadata, &out.Metadata); err != nil {
		return out, err
	}
	out.Document, err = evaldomain.Freeze("DatasetInput", document)
	return out, err
}

// Lists project only the already-safe metadata column; private case documents
// are never loaded while serving collection pages.
func (s *Store) ListDatasets(ctx context.Context, scope Scope, afterID, afterRevision string, limit int) ([]evaldomain.Dataset, error) {
	if limit < 1 || limit > evaldomain.MaxPageSize {
		return nil, evaldomain.Failure("eval_invalid")
	}
	if _, err := s.project(ctx, scope, false); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `SELECT metadata FROM eval_dataset_revisions WHERE owner_id=$1 AND project_id=$2 AND (dataset_id,revision)>($3,$4) ORDER BY dataset_id,revision LIMIT $5`, scope.OwnerID, scope.ProjectID, afterID, afterRevision, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]evaldomain.Dataset, 0)
	for rows.Next() {
		var b []byte
		var d evaldomain.Dataset
		if err = rows.Scan(&b); err != nil {
			return nil, err
		}
		if err = json.Unmarshal(b, &d); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}
