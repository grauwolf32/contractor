package pagelimit

import "testing"

func TestFetchOfTheLargestPageIsTheLargestValidFetch(t *testing.T) {
	if !ValidPage(MaxRows) || ValidPage(MaxRows+1) || ValidPage(0) {
		t.Fatalf("ValidPage accepts the wrong range around %d", MaxRows)
	}
	if FetchRows(MaxRows) != MaxFetchRows || !ValidFetch(FetchRows(MaxRows)) {
		t.Fatalf("FetchRows(%d) = %d is not a valid fetch", MaxRows, FetchRows(MaxRows))
	}
	if ValidFetch(MaxFetchRows+1) || ValidFetch(0) {
		t.Fatalf("ValidFetch accepts the wrong range around %d", MaxFetchRows)
	}
	for limit := 1; limit <= MaxRows; limit++ {
		if !ValidFetch(FetchRows(limit)) {
			t.Fatalf("FetchRows(%d) is not a valid fetch", limit)
		}
	}
}
