package scanplan_test

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// An inventory of 1000 selected operations parses its source and computes the
// source content digest once for preparation (SCN-R256). The inventory's own
// source digest, computed while packaging tasks, is auditdomain work and is
// checked separately against the prepared digest.
func TestLargeInventoryParsesAndDigestsSourceOnce(t *testing.T) {
	for _, key := range []string{"json/sqlmap", "yaml/sqlmap", "json/nuclei"} {
		t.Run(key, func(t *testing.T) {
			format, scanner, _ := strings.Cut(key, "/")
			source, settings := largeSource(t, format), largeSettings(t, scanner)
			work := scanplan.CountSourceWork(t)
			inventory, err := prepareScanBatch(t, source, sourceFormats[format], inventorySourceRef("source-1"), settings)
			if err != nil {
				t.Fatal(err)
			}
			if len(inventory) != largeOperations || work.Parses() != 1 || work.Digests() != 1 {
				t.Fatalf("%d tasks took %d source parses and %d source digests, want 1 and 1", len(inventory), work.Parses(), work.Digests())
			}

		})
	}
}

// The single-operation Run-time path parses the source of each call: there is
// no cache that a later call with the same settings could reuse.
func TestSingleOperationPreparationParsesItsOwnSource(t *testing.T) {
	source := encodeSource(t, reuseDocument(), "json")
	ref := inventorySourceRef("source-1")
	inventory, err := prepareScanBatch(t, source, "application/json", ref, reuseSQLMapSettings)
	if err != nil {
		t.Fatal(err)
	}
	settings, err := compileSelectionFixture([]byte(reuseSQLMapSettings))
	if err != nil {
		t.Fatal(err)
	}
	work := scanplan.CountSourceWork(t)
	for call := 1; call <= 2; call++ {
		if _, err := settings.PrepareOperation(source, "application/json", ref, inventory[0].Operation); err != nil {
			t.Fatal(err)
		}
		if _, err := settings.PrepareOperation(source, "application/json", ref, inventory[1].Operation); err != nil {
			t.Fatal(err)
		}
		if work.Parses() != int64(2*call) || work.Digests() != int64(2*call) {
			t.Fatalf("call %d: %d parses and %d digests", call, work.Parses(), work.Digests())
		}
	}
}

type sourceInput struct {
	name      string
	data      []byte
	mediaType string
	ref       contracts.ArtifactRef
}

// reuseInputs are the same settings' sources that differ only in bytes, media
// type or exact revision; every one must produce its own result.
func reuseInputs(t *testing.T) []sourceInput {
	t.Helper()
	data, yamlData := encodeSource(t, reuseDocument(), "json"), encodeSource(t, reuseDocument(), "yaml")
	return []sourceInput{
		{"json", data, "application/json", inventorySourceRef("source-1")},
		{"json bytes", append(append([]byte{}, data...), ' '), "application/json", inventorySourceRef("source-1")},
		{"json revision", data, "application/json", inventorySourceRef("source-2")},
		{"yaml", yamlData, "application/yaml", inventorySourceRef("source-1")},
		{"yaml media type", yamlData, "text/yaml", inventorySourceRef("source-1")},
	}
}

func collectBatch(settings scanplan.AuditSelection, input sourceInput) ([]scanplan.PreparedAuditOperation, error) {
	results := []scanplan.PreparedAuditOperation{}
	for prepared, err := range settings.PrepareOperations(input.data, input.mediaType, input.ref) {
		if err != nil {
			return nil, err
		}
		results = append(results, prepared)
	}
	return results, nil
}

func TestPrepareOperationsUsesTheSourceOfEachCall(t *testing.T) {
	for _, settingsText := range []string{reuseSQLMapSettings, reuseNucleiSettings} {
		settings, err := compileSelectionFixture([]byte(settingsText))
		if err != nil {
			t.Fatal(err)
		}
		identities := map[string]string{}
		// Revisit the first source last: the settings keep nothing from the
		// sources prepared in between.
		inputs := reuseInputs(t)
		for _, input := range append(inputs, inputs[0]) {
			ref := input.ref.Clone()
			batch, err := collectBatch(settings, sourceInput{input.name, input.data, input.mediaType, ref})
			if err != nil {
				t.Fatalf("%s %s: %v", settings.Scanner(), input.name, err)
			}
			if len(batch) != len(settings.Operations()) {
				t.Fatalf("%s %s: %d results", settings.Scanner(), input.name, len(batch))
			}
			*ref.Revision = "changed-by-caller"
			for index, pointer := range settings.Operations() {
				single, err := settings.PrepareOperation(input.data, input.mediaType, input.ref, pointer)
				if err != nil || !reflect.DeepEqual(batch[index], single) {
					t.Fatalf("%s %s %s: batch result differs from single-operation preparation: %v", settings.Scanner(), input.name, pointer, err)
				}
				source := sourceOf(batch[index])
				if source.ContentDigest != digest(input.data) || !source.Artifact.SameExact(input.ref) {
					t.Fatalf("%s %s %s: result carries another source identity", settings.Scanner(), input.name, pointer)
				}
				if previous, seen := identities[input.name+pointer]; seen && previous != batch[index].PreparationDigest {
					t.Fatalf("%s %s %s: repeated input changed its identity", settings.Scanner(), input.name, pointer)
				}
				identities[input.name+pointer] = batch[index].PreparationDigest
			}
		}
		distinct := map[string]bool{}
		for _, identity := range identities {
			distinct[identity] = true
		}
		if len(distinct) != len(inputs)*len(settings.Operations()) {
			t.Fatalf("%s: %d distinct preparation identities for %d inputs", settings.Scanner(), len(distinct), len(inputs)*len(settings.Operations()))
		}
	}
}

func sourceOf(prepared scanplan.PreparedAuditOperation) (source struct {
	Artifact      contracts.ArtifactRef
	ContentDigest string
}) {
	if prepared.Target != nil {
		source.Artifact, source.ContentDigest = prepared.Target.Source.Artifact, prepared.Target.Source.ContentDigest
	} else {
		source.Artifact, source.ContentDigest = prepared.RequestSet.Source.Artifact, prepared.RequestSet.Source.ContentDigest
	}
	return source
}

// Concurrent batches with the same settings and different sources produce the
// sequential results, and no two results share a source identity's storage.
func TestPrepareOperationsConcurrentBatchesShareNoMutableState(t *testing.T) {
	settings, err := compileSelectionFixture([]byte(reuseSQLMapSettings))
	if err != nil {
		t.Fatal(err)
	}
	inputs := reuseInputs(t)
	expected := make([][]scanplan.PreparedAuditOperation, len(inputs))
	for index, input := range inputs {
		if expected[index], err = collectBatch(settings, input); err != nil {
			t.Fatal(err)
		}
	}
	const rounds = 3
	results := make([][]scanplan.PreparedAuditOperation, rounds*len(inputs))
	failures := make(chan error, len(results))
	var group sync.WaitGroup
	for slot := range results {
		group.Go(func() {
			input := inputs[slot%len(inputs)]
			batch, err := collectBatch(settings, input)
			if err == nil && !reflect.DeepEqual(batch, expected[slot%len(inputs)]) {
				err = fmt.Errorf("%s: concurrent batch differs from the sequential one", input.name)
			}
			results[slot] = batch
			failures <- err
		})
	}
	group.Wait()
	close(failures)
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	first := results[0]
	*first[0].RequestSet.Source.Artifact.Revision = "mutated"
	first[1].RequestSet.Requests[0].Request.URL = "https://mutated.test/"
	first[1].Gaps = append(first[1].Gaps, first[0].Gaps...)
	for slot, batch := range results[1:] {
		if !reflect.DeepEqual(batch, expected[(slot+1)%len(inputs)]) {
			t.Fatal("results of independent batches share storage")
		}
	}
	if *first[1].RequestSet.Source.Artifact.Revision != "source-1" || *expected[0][0].RequestSet.Source.Artifact.Revision != "source-1" {
		t.Fatal("results of one batch share their source revision")
	}
}

// Stopping the sequence early stops preparation; it never prepares the rest.
func TestPrepareOperationsStopsWhenTheCallerStops(t *testing.T) {
	settings, err := compileSelectionFixture([]byte(reuseSQLMapSettings))
	if err != nil {
		t.Fatal(err)
	}
	input := reuseInputs(t)[0]
	visited := 0
	for _, err := range settings.PrepareOperations(input.data, input.mediaType, input.ref) {
		if err != nil {
			t.Fatal(err)
		}
		visited++
		break
	}
	if visited != 1 {
		t.Fatalf("visited %d operations", visited)
	}
}

// BenchmarkLargeScanInventory records allocations and elapsed time of one
// 1000-operation inventory from a source close to 2 MiB as observations.
// Acceptance rests on the work counts and golden parity, not on a latency
// threshold. Reproduce with:
//
//	go test -run '^$' -bench BenchmarkLargeScanInventory -benchtime 3x ./internal/scanplan/
func BenchmarkLargeScanInventory(b *testing.B) {
	for _, format := range []string{"json", "yaml"} {
		b.Run(format, func(b *testing.B) {
			source, settings := largeSource(b, format), largeSettings(b, "sqlmap")
			b.ReportAllocs()
			for b.Loop() {
				if _, err := prepareScanBatch(b, source, sourceFormats[format], inventorySourceRef("source-1"), settings); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
