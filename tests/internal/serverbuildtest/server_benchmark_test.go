package serverbuildtest

import (
	"fmt"
	"path/filepath"
	"testing"
)

// Each iteration models the ordinary process selection's 25 Server requests.
// Run explicitly with -benchtime=1x; these benchmarks invoke the real Go tool.
func BenchmarkServerBuildPerScenario(b *testing.B) {
	root, err := filepath.Abs("../../..")
	if err != nil {
		b.Fatal(err)
	}
	directory := b.TempDir()
	b.ResetTimer()
	for iteration := range b.N {
		for scenario := range 25 {
			output := filepath.Join(directory, fmt.Sprintf("server-%d-%d", iteration, scenario))
			if err := buildServer(root, output, false, "", "", ""); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkServerBuildReuse(b *testing.B) {
	root, err := filepath.Abs("../../..")
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		var cache Cache
		for range 25 {
			if _, err := cache.Binary(root); err != nil {
				_ = cache.Close()
				b.Fatal(err)
			}
		}
		if err := cache.Close(); err != nil {
			b.Fatal(err)
		}
	}
}
