package publicapi

// oapi-codegen currently follows response-component aliases when it builds
// client response wrappers but does not emit Go aliases for those names. Keep
// this small bridge beside generated code so the public OpenAPI can retain its
// readable reusable response names.
type (
	BadRequest           = Error
	Unauthorized         = Error
	Forbidden            = Error
	NotFound             = Error
	Conflict             = Error
	PreconditionFailed   = Error
	PayloadTooLarge      = Error
	UnsupportedMediaType = Error
	UnprocessableEntity  = Error
	InternalError        = Error
)
