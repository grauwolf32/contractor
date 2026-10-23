// Package pagelimit is the one definition of the public list page bound
// (API-R67, LIM-API-18).
//
// A paged public list returns at most MaxRows items (the OpenAPI
// #/components/parameters/Limit maximum). Its handler reads one look-ahead row
// beyond the requested page to decide whether another page follows, so every
// store behind a public paged list accepts a read of up to MaxFetchRows rows.
// Handlers, stores, fakes and the CLI take both numbers from here instead of
// declaring their own.
package pagelimit

const (
	// MaxCursorBytes bounds the shared public continuation cursor (API-R67).
	MaxCursorBytes = 512
	// MaxRows is the largest page a public list returns.
	MaxRows = 200
	// MaxFetchRows is the largest read a store accepts for one public page:
	// MaxRows plus the look-ahead row.
	MaxFetchRows = MaxRows + 1
)

// FetchRows is the number of rows a handler reads for a page of limit rows:
// the page plus the look-ahead row that decides whether a cursor is issued.
func FetchRows(limit int) int { return limit + 1 }

// ValidPage reports whether limit is an acceptable public page size.
func ValidPage(limit int) bool { return limit >= 1 && limit <= MaxRows }

// ValidFetch reports whether a store accepts a read of limit rows for one
// public page, including its look-ahead row.
func ValidFetch(limit int) bool { return limit >= 1 && limit <= MaxFetchRows }
