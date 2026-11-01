package auditservice

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type engagementMaterialResolver interface {
	UsePlaintext(context.Context, string, []runtimesettings.RuntimeCredentialKind, func(runtimesettings.RuntimeCredentialKind, []byte) error) error
}

// MaterializeHTTPIdentities consumes only the trusted allocation projection
// resolved from its Run binding and immutable held Audit baseline. Caller input
// never supplies a credential ID at this boundary. The existing credential
// lifecycle fence protects each synchronous decrypt/copy operation.
func MaterializeHTTPIdentities(ctx context.Context, resolver engagementMaterialResolver, projection *engagementcontract.AllocationProjection) (map[string]runtimesettings.HTTPIdentityMaterial, error) {
	if projection == nil || projection.HTTP == nil {
		return nil, nil
	}
	result := map[string]runtimesettings.HTTPIdentityMaterial{}
	fail := func() (map[string]runtimesettings.HTTPIdentityMaterial, error) {
		for id, material := range result {
			clear(material.Fields)
			delete(result, id)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("engagement private identity material is unavailable")
	}
	for _, identity := range projection.HTTP.Identities {
		if identity.Credential == nil {
			continue
		}
		if resolver == nil {
			return fail()
		}
		ref := *identity.Credential
		err := resolver.UsePlaintext(ctx, ref.CredentialID, []runtimesettings.RuntimeCredentialKind{ref.Kind}, func(kind runtimesettings.RuntimeCredentialKind, data []byte) error {
			material := runtimesettings.HTTPIdentityMaterial{Credential: ref}
			if kind != ref.Kind {
				return ErrInvalid
			}
			if kind == runtimesettings.RuntimeCredentialTargetCookies {
				var body struct {
					Cookies map[string]contracts.SecretString `json:"cookies" strictjson:"required"`
				}
				if strictjson.DecodeNonNull(data, &body) != nil {
					return ErrInvalid
				}
				material.Fields = body.Cookies
			} else {
				if strictjson.DecodeNonNull(data, &material.Fields) != nil {
					return ErrInvalid
				}
			}
			if material.Validate() != nil {
				clear(material.Fields)
				return ErrInvalid
			}
			result[identity.ID] = material
			return nil
		})
		if err != nil {
			return fail()
		}
	}
	if len(result) == 0 {
		return nil, nil
	}
	return result, nil
}
