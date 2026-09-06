package findingintake

import (
	"encoding/json"
	"errors"
	"testing"
)

func collectionRequestFixture() PublishCollectionRequest {
	return PublishCollectionRequest{ClientKey: "snapshot", Sources: []CollectionSelection{
		{Kind: "run", ID: "run", ReceiptIDs: []string{"receipt-b", "receipt-a"}, Findings: []CollectionFindingSelection{}},
		{Kind: "audit", ID: "audit", ReceiptIDs: []string{}, Findings: []CollectionFindingSelection{{FindingID: "finding", Revision: 1}}},
	}}
}

func TestFindingCollectionPublicationRequestIdentity(t *testing.T) {
	input := collectionRequestFixture()
	before, _ := json.Marshal(input)
	normalized, digest, err := canonicalCollectionRequest(input)
	if err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(input)
	if string(before) != string(after) {
		t.Fatal("normalization mutated caller-owned selection")
	}
	_, again, err := canonicalCollectionRequest(normalized)
	if err != nil || digest != again {
		t.Fatalf("selection ordering changed identity: %v", err)
	}
	normalized.Sources[0].Findings[0].Revision++
	_, changed, err := canonicalCollectionRequest(normalized)
	if err != nil || changed == digest {
		t.Fatalf("review revision was omitted from identity: %v", err)
	}
}

func TestFindingCollectionPublicationRequestRejectsAmbiguousSelections(t *testing.T) {
	for name, mutate := range map[string]func(*PublishCollectionRequest){
		"foreign path key":     func(v *PublishCollectionRequest) { v.ClientKey = "../other" },
		"missing sources":      func(v *PublishCollectionRequest) { v.Sources = nil },
		"missing receipts":     func(v *PublishCollectionRequest) { v.Sources[0].ReceiptIDs = nil },
		"missing findings":     func(v *PublishCollectionRequest) { v.Sources[0].Findings = nil },
		"duplicate source":     func(v *PublishCollectionRequest) { v.Sources[1] = v.Sources[0] },
		"duplicate receipt":    func(v *PublishCollectionRequest) { v.Sources[0].ReceiptIDs[1] = v.Sources[0].ReceiptIDs[0] },
		"Run finding selector": func(v *PublishCollectionRequest) { v.Sources[1].Kind = "run" },
		"unversioned finding":  func(v *PublishCollectionRequest) { v.Sources[1].Findings[0].Revision = 0 },
		"unsafe revision":      func(v *PublishCollectionRequest) { v.Sources[1].Findings[0].Revision = 1 << 53 },
		"duplicate finding": func(v *PublishCollectionRequest) {
			v.Sources[1].Findings = append(v.Sources[1].Findings, v.Sources[1].Findings[0])
		},
		"selection limit": func(v *PublishCollectionRequest) { v.Sources[0].ReceiptIDs = make([]string, 257) },
	} {
		t.Run(name, func(t *testing.T) {
			v := collectionRequestFixture()
			mutate(&v)
			if _, _, err := canonicalCollectionRequest(v); !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid selection = %v", err)
			}
		})
	}
}
