package privateartifacts

import (
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

type sitemapCollectionRequest struct {
	APIVersion string                           `json:"apiVersion" strictjson:"required"`
	Input      projectsitemap.CollectionIngress `json:"input" strictjson:"required"`
	Automatic  bool                             `json:"automatic" strictjson:"required"`
}

func sitemapCollector(grant controlplane.AllocationGrant) bool {
	return grant.SitemapPolicy.CollectObservations && grant.WritePolicy == controlplane.WriteInputsAndIntermediates
}
