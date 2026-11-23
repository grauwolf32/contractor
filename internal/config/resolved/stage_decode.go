package resolved

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func (s *ResolvedStage) UnmarshalJSON(data []byte) error {
	type plain ResolvedStage
	var value plain
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := strictjson.Decode(data, &fields); err != nil {
		return err
	}
	if fields["operation"] != nil {
		if value.Operation == nil || fields["planner"] != nil || fields["instructions"] != nil {
			return contracts.Invalidf("operation requires an object and forbids planner and instructions")
		}
	}
	*s = ResolvedStage(value)
	return nil
}
