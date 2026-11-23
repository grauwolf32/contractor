package findingintake

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/jackc/pgx/v5"
)

// ExportReviews keeps Audit authority and full decision reads in their owner,
// using the publisher's repeatable-read transaction rather than another pool.
type ExportReviews interface {
	FindingExportScope(context.Context, pgx.Tx, string, string, bool) (string, error)
	SelectExportFindings(context.Context, pgx.Tx, string, string, []findingexport.Selection) ([]findingexport.Result, error)
}

type PublishedExport struct {
	ExportID      string        `json:"exportId" strictjson:"required"`
	Snapshot      ExactArtifact `json:"snapshot" strictjson:"required"`
	Artifact      ExactArtifact `json:"artifact" strictjson:"required"`
	Diagnostics   ExactArtifact `json:"diagnostics" strictjson:"required"`
	SnapshotAt    string        `json:"snapshotAt" strictjson:"required"`
	SelectedCount int           `json:"selectedCount" strictjson:"required"`
	ResultCount   int           `json:"resultCount" strictjson:"required"`
	WarningCount  int           `json:"warningCount" strictjson:"required"`
	Replayed      bool          `json:"replayed" strictjson:"required"`
}

type ExportDownload struct{ Read artifacts.ReadResult }
