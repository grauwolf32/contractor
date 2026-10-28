package projectsitemap

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestPostgresSitemapSnapshotPreservesSelectedConcreteDeclarations(t *testing.T) {
	for _, structural := range []bool{false, true} {
		for _, selectGroup := range []bool{false, true} {
			t.Run(fmt.Sprintf("structural=%t/group-selection=%t", structural, selectGroup), func(t *testing.T) {
				pool := sitemapPool(t)
				ctx := t.Context()
				createSitemapRun(t, ctx, pool, "snapshot-consumer")
				service := NewService(pool)
				rules := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "rules", Kind: "manual", Mode: "blackbox"}
				manual := rules
				manual.SourceID = "manual"
				restricted := manual
				restricted.SourceID, restricted.Mode = "restricted", "source-assisted"
				revision := "absent"
				sequence := 0
				mutate := func(authority Authority, mutation Mutation) {
					t.Helper()
					sequence++
					mutation.ExpectedRevision = revision
					result, err := service.Mutate(ctx, authority, fmt.Sprintf("declaration-snapshot-%d", sequence), mutation)
					if err != nil {
						t.Fatal(err)
					}
					revision = *result.Ref.Revision
				}
				policyRules := []Rule{
					{SourceID: rules.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{}},
					{SourceID: rules.SourceID, Origin: "https://example.test", Method: "GET", Path: "/other/{id}", Parameters: []string{}},
				}
				selectedURL, otherURL := "https://example.test/users/alice", "https://example.test/other/bob"
				values := []StructuralValue{}
				if structural {
					policyRules[0].Structural = []StructuralParameter{{Name: "action", Allowed: []string{"read", "write"}}}
					selectedURL += "?action=read"
					otherURL = "https://example.test/users/alice?action=write"
					values = []StructuralValue{{Name: "action", Values: []string{"read"}}}
				}
				mutate(rules, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: rules.SourceID, Rules: policyRules}})
				mutate(manual, Mutation{Operation: "declare", URL: selectedURL, Method: "GET"})
				mutate(manual, Mutation{Operation: "declare", URL: otherURL, Method: "GET"})
				mutate(restricted, Mutation{Operation: "declare", URL: selectedURL, Method: "GET"})
				access, err := service.AllocationAccess(ctx, "snapshot-consumer", "research", readerPolicy("blackbox"))
				if err != nil {
					t.Fatal(err)
				}
				member := memberID("https://example.test", "GET", "/users/alice", values...)
				selection := SnapshotSelection{Revision: revision, ExpectedPolicyDigest: access.PolicyDigest, Members: []string{member}}
				if selectGroup {
					selection.Members = nil
					selection.Groups = []string{groupID("https://example.test", "GET", "/users/{id}", values...)}
				}
				result, err := service.SnapshotAllocation(ctx, access, "save", selection)
				if err != nil {
					t.Fatal(err)
				}
				run, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run(access.RunID)
				if err != nil {
					t.Fatal(err)
				}
				stored, err := run.Read(ctx, result.Artifact.Ref)
				if err != nil {
					t.Fatal(err)
				}
				var snapshot snapshotDocument
				if err := json.Unmarshal(stored.Payload.Data, &snapshot); err != nil {
					t.Fatal(err)
				}
				members, declarations := 0, 0
				for _, group := range snapshot.Document.Groups {
					for _, value := range group.Members {
						members++
						if value.ID != member {
							t.Fatal("snapshot added an unselected member")
						}
					}
					for _, declaration := range group.Declarations {
						if declaration.Rule {
							continue
						}
						declarations++
						if declaration.Provenance.SourceID != manual.SourceID || group.ID != groupID("https://example.test", "GET", "/users/alice", values...) {
							t.Fatal("snapshot added an unselected or restricted concrete declaration")
						}
					}
				}
				if members != 1 || declarations != 1 || !slices.ContainsFunc(snapshot.Document.Sources, func(source Source) bool { return source.Provenance.SourceID == manual.SourceID }) {
					t.Fatalf("snapshot lost selected declaration provenance: members=%d declarations=%d", members, declarations)
				}
				if slices.ContainsFunc(snapshot.Document.Sources, func(source Source) bool { return source.Provenance.SourceID == restricted.SourceID }) {
					t.Fatal("snapshot widened the blackbox source policy")
				}
			})
		}
	}
}
