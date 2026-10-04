package findingintake

import "testing"

func TestCollectionReceiptNeedsRetention(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name            string
		receipt         CollectionReceipt
		sourceSucceeded bool
		want            bool
	}{
		{name: "new", receipt: CollectionReceipt{}, want: true},
		{name: "new from succeeded Run", receipt: CollectionReceipt{}, sourceSucceeded: true, want: true},
		{name: "rejected", receipt: CollectionReceipt{Rejected: true}, sourceSucceeded: true},
		{name: "retained from failed Run", receipt: CollectionReceipt{Retained: true}},
		{
			name:    "retained before success, direct assessment pending",
			receipt: CollectionReceipt{Retained: true}, sourceSucceeded: true, want: true,
		},
		{
			name: "retained after the Run finished", sourceSucceeded: true,
			receipt: CollectionReceipt{Retained: true, PostTerminalRetained: true},
		},
		{
			name: "directly assessed", sourceSucceeded: true,
			receipt: CollectionReceipt{Retained: true, DirectAssessed: true},
		},
	} {
		if got := test.receipt.NeedsRetention(test.sourceSucceeded); got != test.want {
			t.Errorf("%s: NeedsRetention(%t) = %t, want %t", test.name, test.sourceSucceeded, got, test.want)
		}
	}
}
