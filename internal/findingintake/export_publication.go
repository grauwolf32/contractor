package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/findingexport/sarif"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const exportPublicationSchema = "contractor.findings.export-publication.v1"

type exportPublication struct {
	Schema        string          `json:"schema" strictjson:"required"`
	RequestDigest string          `json:"request_digest" strictjson:"required"`
	Published     PublishedExport `json:"published" strictjson:"required"`
}

func exportID(auditID, key string) string {
	data, _ := strictjson.Canonical([]string{auditID, key})
	return "export-" + strings.TrimPrefix(contentdigest.Bytes(data), "sha256:")
}

func exportRef(auditID, id, part string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(auditID), Name: "finding-" + part + "." + id}
}

// PublishFindingExport atomically commits one Audit-local snapshot, SARIF,
// diagnostics and frozen publication receipt. A replay consults no live finding.
func (p *CollectionPublisher) PublishFindingExport(ctx context.Context, ownerID, auditID string, input findingexport.Request) (PublishedExport, error) {
	request, digest, err := input.Normalize(auditID)
	if err != nil {
		return PublishedExport{}, err
	}
	if ownerID == "" {
		return PublishedExport{}, ErrInvalid
	}
	reviews, ok := p.reviews.(ExportReviews)
	if !ok {
		return PublishedExport{}, errors.New("finding export review reader is unavailable")
	}
	id := exportID(auditID, request.ClientKey)
	var result PublishedExport
	err = postgres.InTxWithRetry(ctx, p.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		result = PublishedExport{}
		if err := postgres.LockAdvisory(ctx, tx, postgres.NewAdvisoryKey("finding-export", auditID, request.ClientKey)); err != nil {
			return err
		}
		projectID, err := reviews.FindingExportScope(ctx, tx, ownerID, auditID, true)
		if err != nil {
			return err
		}
		scope, err := artifacts.ProjectScope(projectID)
		if err != nil {
			return err
		}
		repository := artifactpostgres.NewRepository(tx)
		previous, _, found, err := readExportPublication(ctx, repository, scope, auditID, id, digest)
		if err != nil || found {
			result = previous
			return err
		}
		rows, err := reviews.SelectExportFindings(ctx, tx, ownerID, auditID, request.Findings)
		if err != nil {
			return err
		}
		var at time.Time
		if err := tx.QueryRow(ctx, `SELECT transaction_timestamp()`).Scan(&at); err != nil {
			return err
		}
		snapshot, err := findingexport.NewSnapshot(auditID, request, at)
		if err != nil {
			return err
		}
		selected := make([]CollectionFindingSelection, len(request.Findings))
		for i, v := range request.Findings {
			selected[i] = CollectionFindingSelection{FindingID: v.FindingID, Revision: v.Revision}
		}
		observer := func(receipt Receipt, proposal auditdomain.FindingProposal, proposalDocument auditdomain.FindingCollectionDocument, documents []auditdomain.FindingCollectionDocument, origins map[string]auditdomain.FindingCollectionDocument, contents map[string][]byte, git map[string]*artifacts.GitSource, analysis *findinganalysis.Resolved) error {
			origin := receipt.Origin
			value := findingexport.Receipt{ReceiptID: receipt.ReceiptID, ProposalID: receipt.ProposalID, Proposal: proposal, Evidence: []findingexport.Evidence{}, Analysis: analysis, Origin: findingexport.Origin{RunID: origin.RunID, InvocationID: origin.InvocationID, StageExecutionID: origin.StageExecutionID, AllocationID: origin.AllocationID, LogicalAgentName: origin.LogicalAgentName, WorkflowName: origin.Workflow.Name, WorkflowVersion: origin.Workflow.Version, WorkflowClosureDigest: origin.Workflow.ClosureDigest}}
			originalProposal := auditdomain.FindingCollectionDocument{Scope: auditdomain.FindingCollectionSource{Kind: "run", ID: origin.RunID}, Ref: receipt.Proposal.Ref, Digest: receipt.Proposal.Digest, MediaType: receipt.Proposal.MediaType, SizeBytes: receipt.Proposal.SizeBytes}
			var err error
			originalProposal.ID, err = auditdomain.FindingCollectionDocumentID(originalProposal)
			if err != nil {
				return err
			}
			value.ProposalArtifact = findingexport.Evidence{Original: originalProposal, Retained: proposalDocument}
			if origin.Audit != nil {
				value.Origin.AuditID = origin.Audit.AuditID
				value.Origin.ExecutionID = origin.Audit.ExecutionID
				value.Origin.Role = origin.Audit.Role
			}
			for _, doc := range documents {
				original := origins[doc.ID]
				evidence := findingexport.Evidence{Original: original, Retained: doc}
				if source := git[doc.ID]; source != nil {
					evidence.Git = &findingexport.GitSource{RepositoryURI: source.RepositoryURL, Commit: source.ResolvedCommit}
				}
				value.Evidence = append(value.Evidence, evidence)
				if analysis == nil && original.MediaType == auditdomain.HTTPExchangeMediaType && original.Ref.Namespace == artifactpolicy.FindingProposalNamespace && original.Ref.Name == "http.exchange."+receipt.ProposalID {
					projected, err := findinganalysis.ProjectHTTP(findinganalysis.HTTPExchange{ID: "selected-exchange", EvidenceRef: original.Ref, Decoder: "contractor-http-exchange/v1"}, original, contents[doc.ID])
					if err != nil {
						return err
					}
					value.HTTP = &projected
				}
			}
			if analysis != nil && analysis.Analysis.PrimaryHTTPExchangeID != "" {
				for _, h := range analysis.HTTP {
					if h.ID == analysis.Analysis.PrimaryHTTPExchangeID {
						copy := h
						value.HTTP = &copy
						break
					}
				}
			}
			if value.HTTP == nil && analysis == nil && proposal.HTTPExchange != nil {
				last := proposal.HTTPExchange.Attempts[len(proposal.HTTPExchange.Attempts)-1]
				value.HTTP = &findinganalysis.ResolvedHTTP{ID: "proposal-exchange", Method: last.Method, URI: findingexport.SanitizedURI(last.URL), Status: last.Status, Origin: "producer-asserted", Omitted: []string{"bodies", "headers", "query", "fragment", "userinfo"}}
			}
			return snapshot.AddReceipt(value)
		}
		if _, _, err := p.captureCollection(ctx, tx, ownerID, PublishCollectionRequest{ClientKey: request.ClientKey, Sources: []CollectionSelection{{Kind: "audit", ID: auditID, ReceiptIDs: []string{}, Findings: selected}}}, at, observer); err != nil {
			return err
		}
		for _, row := range rows {
			if err := snapshot.AddResult(row); err != nil {
				return err
			}
		}
		frozen, err := findingexport.EncodeSnapshot(snapshot)
		if err != nil {
			return err
		}
		rendered, diagnostics, err := sarif.Render(snapshot)
		if err != nil {
			return err
		}
		diagnosticBytes, err := findingexport.EncodeDiagnostics(diagnostics)
		if err != nil {
			return err
		}
		service := artifacts.NewService(repository)
		write := func(part string, data []byte) (ExactArtifact, error) {
			written, err := service.CreateFrozen(ctx, scope, exportRef(auditID, id, part), artifacts.Payload{MediaType: "application/json", Data: data})
			if err != nil {
				return ExactArtifact{}, err
			}
			return ExactArtifact{Ref: written.Ref, Digest: contentdigest.Bytes(data), MediaType: written.MediaType, SizeBytes: written.Size}, nil
		}
		groups, err := snapshot.ResultGroups()
		if err != nil {
			return err
		}
		result = PublishedExport{ExportID: id, SnapshotAt: snapshot.SnapshotAt, SelectedCount: len(request.Findings), ResultCount: len(groups), WarningCount: len(diagnostics.Items)}
		if result.Snapshot, err = write("export-snapshot", frozen); err != nil {
			return err
		}
		if result.Artifact, err = write("export", rendered); err != nil {
			return err
		}
		if result.Diagnostics, err = write("export-diagnostics", diagnosticBytes); err != nil {
			return err
		}
		encoded, err := json.Marshal(exportPublication{Schema: exportPublicationSchema, RequestDigest: digest, Published: result})
		if err != nil {
			return err
		}
		if _, err := write("export-receipt", encoded); err != nil {
			return err
		}
		return ctx.Err()
	})
	if err != nil {
		return PublishedExport{}, err
	}
	return result, nil
}

func (p *CollectionPublisher) ReadFindingExport(ctx context.Context, ownerID, auditID, id, part string) (ExportDownload, error) {
	if part != "export" && part != "export-snapshot" && part != "export-diagnostics" {
		return ExportDownload{}, ErrInvalid
	}
	if contracts.ValidateResourceID("audit", auditID) != nil || len(id) != 71 || !strings.HasPrefix(id, "export-") || !contentdigest.Valid("sha256:"+strings.TrimPrefix(id, "export-")) {
		return ExportDownload{}, ErrInvalid
	}
	reviews, ok := p.reviews.(ExportReviews)
	if !ok {
		return ExportDownload{}, errors.New("finding export review reader is unavailable")
	}
	var result ExportDownload
	err := postgres.InTxWithRetry(ctx, p.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		projectID, err := reviews.FindingExportScope(ctx, tx, ownerID, auditID, false)
		if err != nil {
			return err
		}
		scope, err := artifacts.ProjectScope(projectID)
		if err != nil {
			return err
		}
		_, contents, found, err := readExportPublication(ctx, artifactpostgres.NewRepository(tx), scope, auditID, id, "")
		if err != nil {
			return err
		}
		if !found {
			return ErrNotFound
		}
		result = ExportDownload{Read: contents[part]}
		return nil
	})
	return result, err
}
