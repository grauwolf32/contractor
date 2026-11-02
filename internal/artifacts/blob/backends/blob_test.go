package backends

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
)

func TestBlobConfigSelectionModes(t *testing.T) {
	for _, tc := range []struct {
		kind, path string
		valid      bool
	}{
		{"", "", true}, {"postgresql", "", true}, {"filesystem", "/var/blobs", true},
		{"postgresql", "/var/blobs", true}, {"filesystem", "", false},
		{"filesystem", "relative", false}, {"filesystem", "/var/../blobs", false},
		{"s3", "", false}, {"auto", "", true}, {"auto", "/var/blobs", true},
		{"auto", "relative", false}, {"auto", "/var/../blobs", false},
	} {
		_, err := blob.ValidateConfig(tc.kind, tc.path)
		if (err == nil) != tc.valid {
			t.Errorf("config %q %q: %v", tc.kind, tc.path, err)
		}
	}
}

func TestInlineBlobVerifiesContentAndCancellation(t *testing.T) {
	s := inline.Store{}
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
		if _, err := s.Read(context.Background(), object); !errors.Is(err, blob.ErrIntegrity) {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Store(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
