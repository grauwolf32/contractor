package privateartifacts

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

var (
	errSitemapAccessDenied = errors.New("allocation does not permit this sitemap operation")
	errSitemapUnavailable  = errors.New("sitemap service is unavailable")
)

type sitemapMutationRequest struct {
	APIVersion string                        `json:"apiVersion" strictjson:"required"`
	Mutation   projectsitemap.EditorMutation `json:"mutation" strictjson:"required"`
}
type sitemapSnapshotRequest struct {
	APIVersion string                           `json:"apiVersion" strictjson:"required"`
	Selection  projectsitemap.SnapshotSelection `json:"selection" strictjson:"required"`
}
