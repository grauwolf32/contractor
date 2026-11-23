// Package zipdirectory bounds an untrusted ZIP central directory before
// archive/zip allocates one zip.File per record.
package zipdirectory

import (
	"encoding/binary"
	"errors"
)

// MaximumDirectoryBytes bounds the central directory itself.
const MaximumDirectoryBytes = 4 << 20

var (
	ErrInvalid = errors.New("ZIP central directory is invalid or unsupported")
	ErrLimit   = errors.New("ZIP central directory exceeds its limit")
)

// Check counts the actual central-directory records, never trusting the
// advertised entry count, and returns the offset where the directory starts.
// ZIP64 directories, split archives and nonstandard trailing data are
// rejected; bounded archives fit classic ZIP.
func Check(data []byte, maximumEntries int) (int, error) {
	const endSize = 22
	end := -1
	for i := len(data) - endSize; i >= max(0, len(data)-endSize-65535); i-- {
		if binary.LittleEndian.Uint32(data[i:]) == 0x06054b50 {
			// Do not skip a later signature with an invalid comment length:
			// archive/zip could select it instead of the record we validated.
			if i+endSize+int(binary.LittleEndian.Uint16(data[i+20:])) != len(data) {
				return 0, ErrInvalid
			}
			end = i
			break
		}
	}
	if end < 0 {
		return 0, ErrInvalid
	}
	// Never let archive/zip switch to a ZIP64 directory outside this preflight,
	// including a locator hidden inside the final central-directory comment.
	if end >= 20 && binary.LittleEndian.Uint32(data[end-20:]) == 0x07064b50 {
		return 0, ErrInvalid
	}
	header := data[end:]
	u16 := binary.LittleEndian.Uint16
	u32 := binary.LittleEndian.Uint32
	count := int(u16(header[10:]))
	if u16(header[4:]) != 0 || u16(header[6:]) != 0 || int(u16(header[8:])) != count {
		return 0, ErrInvalid
	}
	if count > maximumEntries {
		return 0, ErrLimit
	}
	size, start := uint64(u32(header[12:])), uint64(u32(header[16:]))
	if start+size != uint64(end) {
		return 0, ErrInvalid
	}
	if size > MaximumDirectoryBytes {
		return 0, ErrLimit
	}
	pos, actual := int(start), 0
	for pos < end {
		if end-pos < 46 || u32(data[pos:]) != 0x02014b50 || u16(data[pos+34:]) != 0 {
			return 0, ErrInvalid
		}
		length := 46 + int(u16(data[pos+28:])) + int(u16(data[pos+30:])) + int(u16(data[pos+32:]))
		if length > end-pos {
			return 0, ErrInvalid
		}
		pos += length
		actual++
		if actual > maximumEntries {
			return 0, ErrLimit
		}
	}
	if actual != count {
		return 0, ErrInvalid
	}
	return int(start), nil
}
