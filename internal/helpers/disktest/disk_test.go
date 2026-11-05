package disktest

import (
	"fmt"
	"strings"
	"testing"
)

type stopped struct{}

type recordingTB struct {
	testing.TB
	outcome string
	message string
}

func (*recordingTB) Helper() {}
func (t *recordingTB) Skip(args ...any) {
	t.outcome, t.message = "skip", fmt.Sprint(args...)
	panic(stopped{})
}
func (t *recordingTB) Fatal(args ...any) {
	t.outcome, t.message = "fail", fmt.Sprint(args...)
	panic(stopped{})
}

func TestRequireExplicitOptIn(t *testing.T) {
	for _, test := range []struct{ value, outcome string }{
		{"", "skip"}, {"0", "skip"}, {"1", "run"}, {"true", "fail"}, {"yes", "fail"},
	} {
		t.Run(test.value, func(t *testing.T) {
			t.Setenv("CONTRACTOR_TEST_HEAVY_IO", test.value)
			recorded := &recordingTB{}
			func() {
				defer func() {
					if value := recover(); value != nil {
						if _, ok := value.(stopped); !ok {
							panic(value)
						}
					}
				}()
				Require(recorded)
				recorded.outcome = "run"
			}()
			if recorded.outcome != test.outcome {
				t.Fatalf("outcome = %s, want %s", recorded.outcome, test.outcome)
			}
			if test.outcome == "skip" && !strings.Contains(recorded.message, "CONTRACTOR_TEST_HEAVY_IO=1") {
				t.Fatal("skip omitted the explicit opt-in")
			}
		})
	}
}
