package core

import (
	"errors"
	"testing"
)

func TestInPackage(t *testing.T) {
	TestOnly()
	if Recursive(2) != 0 || !errors.Is(ErrOnlyTests, ErrOnlyTests) {
		t.Fatal("unexpected result")
	}
	if NewStore().TestHook() == "" || (Hidden{}).Self() != (Hidden{}) {
		t.Fatal("unexpected store")
	}
	NewReader().Close()
	if CodeB == CodeA || NewBox(1).Peek() != 1 {
		t.Fatal("unexpected code")
	}
	var asserted Asserted = Impl{}
	_ = asserted
}
