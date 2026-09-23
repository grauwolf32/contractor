package artifactpreview

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/zipdirectory"
)

// checkDirectory bounds actual central-directory records before zip.NewReader
// can allocate one zip.File per record. ZIP64 directories, split archives and
// nonstandard trailing data are deliberately download-only; all supported
// artifacts fit classic ZIP.
func checkDirectory(data []byte) (int, error) {
	offset, err := zipdirectory.Check(data, MaximumEntries)
	if errors.Is(err, zipdirectory.ErrLimit) {
		return 0, ErrLimit
	}
	if err != nil {
		return 0, ErrInvalid
	}
	return offset, nil
}
