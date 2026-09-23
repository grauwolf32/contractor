package public

import (
	"reflect"
	"testing"
)

func TestPaginateTrimsTheExtraRowAndSignsTheLastKeptRow(t *testing.T) {
	h := pinnedCursorHandler()
	position := func(last string) []string { return []string{last} }

	rows, page, err := paginate(h, []string{"a", "b"}, 2, "letters", position)
	if err != nil || !reflect.DeepEqual(rows, []string{"a", "b"}) || page.HasMore || page.NextCursor != nil {
		t.Fatalf("paginate(last page) = %v, %+v, %v", rows, page, err)
	}

	rows, page, err = paginate(h, []string{"a", "b", "c"}, 2, "letters", position)
	if err != nil || !reflect.DeepEqual(rows, []string{"a", "b"}) || !page.HasMore || page.NextCursor == nil {
		t.Fatalf("paginate(more) = %v, %+v, %v", rows, page, err)
	}
	values, err := h.decodePageCursor(*page.NextCursor, "letters", 1)
	if err != nil || !reflect.DeepEqual(values, []string{"b"}) {
		t.Fatalf("decodePageCursor(next) = %v, %v", values, err)
	}
}
