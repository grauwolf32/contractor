package public

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// API-R70: a request body is decoded strictly, including inside request types
// that decode their own members, so a case alias or a duplicate key is
// rejected instead of being decoded into the field it resembles.
func TestRequestBodiesRejectCaseAliasesAndDuplicateKeys(t *testing.T) {
	for _, test := range []struct {
		name    string
		target  func() any
		valid   string
		invalid []string
	}{
		{
			name:   "Project",
			target: func() any { return new(createProjectRequest) },
			valid:  `{"kind":"project","name":"p","description":"d"}`,
			invalid: []string{
				`{"kind":"project","name":"p","Description":"d"}`,
				`{"kind":"project","NAME":"p"}`,
				`{"kind":"project","name":"a","name":"b"}`,
			},
		},
		{
			name:   "Runtime credential material",
			target: func() any { return new(createRuntimeCredentialRequest) },
			valid:  `{"credentialId":"c","kind":"http-proxy-bearer@1","material":{"token":"t"}}`,
			invalid: []string{
				`{"credentialId":"c","Kind":"http-proxy-bearer@1","material":{"token":"t"}}`,
				`{"credentialId":"c","kind":"http-proxy-bearer@1","material":{"Token":"t"}}`,
				`{"credentialId":"c","kind":"http-proxy-bearer@1","material":{"token":"t","token":"u"}}`,
				`{"credentialId":"c","credentialId":"d","kind":"http-proxy-bearer@1","material":{"token":"t"}}`,
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			decode := func(body string) error {
				request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
				return decodeJSON(httptest.NewRecorder(), request, test.target())
			}
			if err := decode(test.valid); err != nil {
				t.Fatalf("valid body rejected: %v", err)
			}
			for _, body := range test.invalid {
				if err := decode(body); !errors.Is(err, errInvalidRequest) {
					t.Errorf("decodeJSON(%s) = %v, want invalid request", body, err)
				}
			}
		})
	}
}
