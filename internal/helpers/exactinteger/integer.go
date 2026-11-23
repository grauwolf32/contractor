// Package exactinteger owns the fixed integer range shared by JSON consumers
// that carry numbers as IEEE 754 binary64 values.
package exactinteger

// Max is the largest integer in binary64's contiguous exact integer range.
// This is a format property, not a configurable application limit.
const Max = 1<<53 - 1
