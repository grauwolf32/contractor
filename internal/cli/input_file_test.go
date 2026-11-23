package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenRegularInputRejectsNonRegularPaths(t *testing.T) {
	directory := t.TempDir()
	regular := filepath.Join(directory, "request.json")
	if err := os.WriteFile(regular, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := openRegularInput(regular, "Run request")
	if err != nil {
		t.Fatal(err)
	}
	file.Close()

	link := filepath.Join(directory, "link.json")
	if err := os.Symlink(regular, link); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{link, directory} {
		if _, err := openRegularInput(path, "Run request"); err == nil ||
			!strings.Contains(err.Error(), "Run request must be a regular file") {
			t.Fatalf("openRegularInput(%q) error = %v", path, err)
		}
	}
}
