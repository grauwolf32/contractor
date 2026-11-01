package auditdomain

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedRoundResultsStrictCompleteDocuments(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "audit-composition", "round-results-cases.json")
	var fixture struct {
		Description string          `json:"description"`
		Limits      map[string]int  `json:"limits"`
		Page        json.RawMessage `json:"page"`
		Manifest    json.RawMessage `json:"manifest"`
		Consumer    json.RawMessage `json:"consumer"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]int{"documentBytes": MaximumRoundResultsDocumentBytes, "pages": MaximumRoundResultsPages, "totalBytes": MaximumRoundResultsBytes, "pageItems": MaximumRoundResultsPageItems, "sliceDescriptors": MaximumRoundResultsSliceDescriptors, "sliceBytes": MaximumRoundResultsSliceBytes} {
		if fixture.Limits[name] != want {
			t.Fatalf("shared %s=%d,want%d", name, fixture.Limits[name], want)
		}
	}
	page, err := DecodeRoundResultsPage(fixture.Page)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := DecodeRoundResultsManifest(fixture.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	consumer, err := decodeRoundResultsConsumer(fixture.Consumer)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := EncodeRoundResultsPage(page)
	if err != nil {
		t.Fatal(err)
	}
	if DigestBytes(canonical) != manifest.Pages[0].Artifact.Digest || int64(len(canonical)) != manifest.MetadataBytes {
		t.Fatal("shared manifest does not pin exact canonical page bytes")
	}
	if page.Items[0].SelectedAttempt.Attempt != 2 || page.Items[0].Findings[0].State != "proposed" || consumer.SourceGeneration != manifest.SourceGeneration {
		t.Fatal("attempt, conservative finding state or source generation lost")
	}
	for name, mutate := range map[string]func(*RoundResultsConsumer){
		"missing mapping":      func(c *RoundResultsConsumer) { c.Mappings = nil },
		"missing member":       func(c *RoundResultsConsumer) { c.ItemCount++ },
		"empty page":           func(c *RoundResultsConsumer) { c.Pages[0].ItemCount = 0 },
		"wrong media":          func(c *RoundResultsConsumer) { c.Pages[0].Artifact.MediaType = "text/plain" },
		"different local page": func(c *RoundResultsConsumer) { c.Mappings[0].Consumer.Ref.Name = "other-page" },
	} {
		fresh, _ := decodeRoundResultsConsumer(fixture.Consumer)
		mutate(&fresh)
		if _, err := EncodeRoundResultsConsumer(fresh); err == nil {
			t.Errorf("consumer accepts %s", name)
		}
	}
	for name, decode := range map[string]func([]byte) error{
		"page":     func(data []byte) error { _, err := DecodeRoundResultsPage(data); return err },
		"manifest": func(data []byte) error { _, err := DecodeRoundResultsManifest(data); return err },
		"consumer": func(data []byte) error { _, err := decodeRoundResultsConsumer(data); return err },
	} {
		valid := map[string][]byte{"page": fixture.Page, "manifest": fixture.Manifest, "consumer": fixture.Consumer}[name]
		for label, bad := range map[string][]byte{
			"unknown":   append([]byte(`{"unknown":true,`), bytes.TrimSpace(valid)[1:]...),
			"alias":     bytes.Replace(valid, []byte(`"schema"`), []byte(`"Schema"`), 1),
			"duplicate": append([]byte(`{"schema":"duplicate",`), bytes.TrimSpace(valid)[1:]...),
			"trailing":  append(append([]byte{}, valid...), []byte(` {}`)...),
			"oversize":  bytes.Repeat([]byte(" "), MaximumRoundResultsDocumentBytes+1),
		} {
			if err := decode(bad); err == nil {
				t.Errorf("%s accepts %s", name, label)
			}
		}
	}
	for label, mutate := range map[string]func(*RoundResultsPage){
		"duplicate item":             func(p *RoundResultsPage) { p.Items = append(p.Items, p.Items[0]) },
		"absent accepted result":     func(p *RoundResultsPage) { p.Items[0].Result = nil },
		"missing receipt":            func(p *RoundResultsPage) { p.Items[0].SelectedAttempt.ReceiptID = "" },
		"false semantic disposition": func(p *RoundResultsPage) { p.Items[0].FinalDisposition = "supported" },
		"unknown source class":       func(p *RoundResultsPage) { p.Items[0].Attachments[0].Classification = "permission-grant" },
		"page ceiling":               func(p *RoundResultsPage) { p.Ordinal = MaximumRoundResultsPages },
		"one oversized item": func(p *RoundResultsPage) {
			p.Items[0].Coverage.Rationale = strings.Repeat("x", MaximumRoundResultsDocumentBytes)
		},
	} {
		fresh, _ := DecodeRoundResultsPage(fixture.Page)
		mutate(&fresh)
		if _, err := EncodeRoundResultsPage(fresh); err == nil {
			t.Errorf("accepted %s", label)
		}
	}
	manifest.Pages = append(manifest.Pages, manifest.Pages[0])
	if _, err := EncodeRoundResultsManifest(manifest); err == nil {
		t.Fatal("manifest accepts repeated/misordered page")
	}
}

func decodeRoundResultsConsumer(data []byte) (RoundResultsConsumer, error) {
	var value RoundResultsConsumer
	if len(data) > MaximumRoundResultsDocumentBytes || strictjson.Decode(data, &value) != nil {
		return value, invalid(CodeInvalid, "round_results_consumer")
	}
	_, err := EncodeRoundResultsConsumer(value)
	return value, err
}
