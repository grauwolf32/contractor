package runtimesettings

// The allocation-wire encoding of RuntimeSettings. contracts.SecretString
// redacts itself under default JSON encoding (SEC-R88); RevealJSON is the one
// encoder that writes the plaintext, for the prepare request the Control Plane
// sends to the Runtime Agent over mTLS (SEC-R29). Each revealed form embeds
// the settings type and shadows only its secret fields with plaintext ones, so
// every other field keeps the default encoding without a second field list.

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// RevealJSON encodes the settings with every SecretString in plaintext. Only
// the allocation prepare request may carry its result; any other encoding of
// RuntimeSettings uses json.Marshal, which redacts the same fields.
func (s RuntimeSettings) RevealJSON() ([]byte, error) {
	type plain RuntimeSettings
	return json.Marshal(struct {
		plain
		LLMGatewayToken  *string                   `json:"llmGatewayToken,omitempty"`
		Telemetry        *revealedTelemetry        `json:"telemetry,omitempty"`
		HTTPProxy        *revealedHTTPProxy        `json:"httpProxy,omitempty"`
		Caido            *revealedCaido            `json:"caido,omitempty"`
		HTTPOriginTarget *revealedHTTPOriginTarget `json:"httpOriginTarget,omitempty"`
	}{
		plain:            plain(s),
		LLMGatewayToken:  revealSecret(s.LLMGatewayToken),
		Telemetry:        revealTelemetry(s.Telemetry),
		HTTPProxy:        revealHTTPProxy(s.HTTPProxy),
		Caido:            revealCaido(s.Caido),
		HTTPOriginTarget: revealHTTPOriginTarget(s.HTTPOriginTarget),
	})
}

type plainTelemetry TelemetrySettings

type revealedTelemetry struct {
	plainTelemetry
	Headers map[string]string `json:"headers"`
}

type plainHTTPProxy HTTPProxySettings

type revealedHTTPProxy struct {
	plainHTTPProxy
	BasicAuth   *revealedBasicAuth `json:"basicAuth,omitempty"`
	BearerToken *string            `json:"bearerToken,omitempty"`
}

type plainCaido CaidoSettings

type revealedCaido struct {
	plainCaido
	BearerToken *string `json:"bearerToken,omitempty"`
}

type plainHTTPOriginTarget HTTPOriginTargetSettings

type revealedHTTPOriginTarget struct {
	plainHTTPOriginTarget
	BasicAuth   *revealedBasicAuth `json:"basicAuth,omitempty"`
	BearerToken *string            `json:"bearerToken,omitempty"`
}

type revealedBasicAuth struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func revealSecret(secret *contracts.SecretString) *string {
	if secret == nil {
		return nil
	}
	value := secret.Reveal()
	return &value
}

func revealBasicAuth(auth *HTTPProxyBasicAuth) *revealedBasicAuth {
	if auth == nil {
		return nil
	}
	return &revealedBasicAuth{Username: auth.Username.Reveal(), Password: auth.Password.Reveal()}
}

func revealTelemetry(settings *TelemetrySettings) *revealedTelemetry {
	if settings == nil {
		return nil
	}
	result := &revealedTelemetry{plainTelemetry: plainTelemetry(*settings)}
	if settings.Headers != nil {
		result.Headers = make(map[string]string, len(settings.Headers))
		for name, value := range settings.Headers {
			result.Headers[name] = value.Reveal()
		}
	}
	return result
}

func revealHTTPProxy(settings *HTTPProxySettings) *revealedHTTPProxy {
	if settings == nil {
		return nil
	}
	return &revealedHTTPProxy{
		plainHTTPProxy: plainHTTPProxy(*settings),
		BasicAuth:      revealBasicAuth(settings.BasicAuth),
		BearerToken:    revealSecret(settings.BearerToken),
	}
}

func revealCaido(settings *CaidoSettings) *revealedCaido {
	if settings == nil {
		return nil
	}
	return &revealedCaido{plainCaido: plainCaido(*settings), BearerToken: revealSecret(settings.BearerToken)}
}

func revealHTTPOriginTarget(settings *HTTPOriginTargetSettings) *revealedHTTPOriginTarget {
	if settings == nil {
		return nil
	}
	return &revealedHTTPOriginTarget{
		plainHTTPOriginTarget: plainHTTPOriginTarget(*settings),
		BasicAuth:             revealBasicAuth(settings.BasicAuth),
		BearerToken:           revealSecret(settings.BearerToken),
	}
}
