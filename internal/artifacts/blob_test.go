package artifacts

import (
	"context"
	"errors"
	"testing"
)

func TestBlobConfigIsExplicit(t *testing.T) {
	for _, tc := range []struct {
		kind, path string
		valid      bool
	}{
		{"", "", true}, {"postgresql", "", true}, {"filesystem", "/var/blobs", true},
		{"postgresql", "/var/blobs", false}, {"filesystem", "", false},
		{"filesystem", "relative", false}, {"filesystem", "/var/../blobs", false},
		{"s3", "", false}, {"auto", "", false},
	} {
		_, err := ValidateBlobConfig(tc.kind, tc.path)
		if (err == nil) != tc.valid {
			t.Errorf("config %q %q: %v", tc.kind, tc.path, err)
		}
	}
}

func TestInlineBlobVerifiesContentAndCancellation(t *testing.T) {
	s := PostgresBlobStore{}
	for _, data := range [][]byte{nil, []byte("original")} {
		object, err := s.Store(context.Background(), data)
		if err != nil {
			t.Fatal(err)
		}
		got, err := s.Read(context.Background(), object)
		if err != nil || string(got) != string(data) {
			t.Fatalf("round trip: %v", err)
		}
		object.Size++
		if _, err := s.Read(context.Background(), object); !errors.Is(err, ErrArtifactIntegrity) {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Store(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
