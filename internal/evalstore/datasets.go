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
	return mutate(ctx, s, scope, "datasets", "dataset-create", id, func() (DatasetReceipt, error) {
		metadata, err := evaldomain.DatasetProjection(input, scope.ProjectID, revision)
		if err != nil {
			return DatasetReceipt{}, err
		}
		_, err = s.db.Exec(ctx, `INSERT INTO eval_dataset_revisions(owner_id,project_id,dataset_id,revision,document,metadata) VALUES($1,$2,$3,$4,$5,$6)`, scope.OwnerID, scope.ProjectID, input.DatasetID, revision, document.Bytes(), bytesOf(metadata))
		return DatasetReceipt{DatasetID: input.DatasetID, DatasetRevision: revision}, normalize(err)
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
