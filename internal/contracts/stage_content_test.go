package contracts

import (
	"strings"
	"testing"
	"time"
)

// The schema requires parameters, artifacts and the deadline, so a nil map,
// which encodes to null, and a zero deadline are invalid both when the
// Planner builds a request and on decode (WRK-R6).
func TestStageContentRequestRequiresMapsAndDeadline(t *testing.T) {
	valid := StageContentRequest{
		APIVersion: APIVersion, SubtaskID: "0", Objective: "check", Instructions: "inspect",
		Parameters: map[string]string{}, Artifacts: map[string]ArtifactRef{},
		Deadline: time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC),
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("empty maps rejected: %v", err)
	}
	withoutParameters, withoutArtifacts, withoutDeadline := valid, valid, valid
	withoutParameters.Parameters = nil
	withoutArtifacts.Artifacts = nil
	withoutDeadline.Deadline = time.Time{}
	for name, request := range map[string]StageContentRequest{
		"nil parameters": withoutParameters, "nil artifacts": withoutArtifacts, "zero deadline": withoutDeadline,
	} {
		if err := request.Validate(); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	const prefix = `{"apiVersion":"contractor/v1alpha1","subtaskId":"0","objective":"check","instructions":"inspect"`
	for _, input := range []string{
		prefix + `,"artifacts":{},"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":{},"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":null,"artifacts":{},"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":{},"artifacts":null,"deadline":"2026-10-11T12:00:00Z"}`,
		prefix + `,"parameters":{},"artifacts":{}}`,
		prefix + `,"parameters":{},"artifacts":{},"deadline":null}`,
		prefix + `,"parameters":{},"artifacts":{},"deadline":"0001-01-01T00:00:00Z"}`,
		prefix + `,"parameters":{},"artifacts":{},"deadline":"2026-10-11T12:00:00"}`,
	} {
		if _, err := DecodePrivateStrict[StageContentRequest]([]byte(input)); err == nil {
			t.Errorf("StageContentRequest accepted: %s", input)
		}
	}
	decoded, err := DecodePrivateStrict[StageContentRequest]([]byte(prefix + `,"parameters":{},"artifacts":{},"deadline":"2026-10-11T14:00:00+02:00"}`))
	if err != nil {
		t.Fatalf("valid StageContentRequest rejected: %v", err)
	}
	if !decoded.Deadline.Equal(valid.Deadline) {
		t.Fatalf("deadline = %s, want %s", decoded.Deadline, valid.Deadline)
	}
}

func TestStageContentRequestValidateCountsCompactUTF8(t *testing.T) {
	deadline := time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC)
	request := func(instructions string) StageContentRequest {
		return StageContentRequest{
			APIVersion: APIVersion, SubtaskID: "0", Objective: "check", Instructions: instructions,
			Parameters: map[string]string{}, Artifacts: map[string]ArtifactRef{}, Deadline: deadline,
		}
	}
	// 64 Ki '<' encode to 384 KiB with HTML escaping but 64 KiB compactly.
	if err := request(strings.Repeat("<", 64*1024)).Validate(); err != nil {
		t.Fatalf("HTML instructions within the compact bound rejected: %v", err)
	}
	if err := request(strings.Repeat("x", MaxStageRequestBytes)).Validate(); err == nil {
		t.Fatal("request over its encoded bound accepted")
	}
}
