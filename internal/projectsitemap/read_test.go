package projectsitemap

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

type sitemapTestView struct {
	Revision string
	Document Document
}

// readSitemap inspects stored results with ordinary Artifact reads and the
// domain decoder. API authorization is tested at the public/private boundary.
func readSitemap(s *Service, ctx context.Context, projectID, revision string) (sitemapTestView, error) {
	store, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Project(projectID)
	if err != nil {
		return sitemapTestView{}, err
	}
	ref := mapRef()
	if revision != "" {
		ref.Revision = &revision
	}
	read, err := store.Read(ctx, ref)
	if errors.Is(err, artifacts.ErrArtifactNotFound) && revision == "" {
		return sitemapTestView{Document: Empty()}, nil
	}
	if err != nil {
		return sitemapTestView{}, err
	}
	if read.Payload.MediaType != MediaType {
		return sitemapTestView{}, ErrInvalid
	}
	document, err := Decode(read.Payload.Data)
	if err != nil {
		return sitemapTestView{}, err
	}
	return sitemapTestView{Revision: *read.Ref.Revision, Document: document}, nil
}
