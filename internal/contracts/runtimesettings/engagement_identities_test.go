package runtimesettings

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestEngagementPrivateMaterialPreservesCredentialKindGrammar(t *testing.T) {
	for _, tc := range []struct {
		kind   RuntimeCredentialKind
		fields map[string]string
		valid  bool
	}{
		{RuntimeCredentialTargetBasic, map[string]string{"username": "пользователь", "password": " пароль "}, true},
		{RuntimeCredentialTargetCookies, map[string]string{"session": "token"}, true},
		{RuntimeCredentialTargetCookies, map[string]string{"session": "pass\tword"}, false},
		{RuntimeCredentialTargetBearer, map[string]string{"token": "part\tpart"}, true},
		{RuntimeCredentialTargetBearer, map[string]string{"token": " token "}, false},
		{RuntimeCredentialTargetBearer, map[string]string{"token": "секрет"}, false},
	} {
		material := HTTPIdentityMaterial{Credential: RuntimeCredentialRef{CredentialID: "held-target", Kind: tc.kind}, Fields: map[string]contracts.SecretString{}}
		for key, value := range tc.fields {
			material.Fields[key] = contracts.NewSecretString(value)
		}
		if (material.Validate() == nil) != tc.valid {
			t.Fatalf("kind=%s accepted=%v want=%v", tc.kind, material.Validate() == nil, tc.valid)
		}
	}
}
