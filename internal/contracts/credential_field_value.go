package contracts

// ValidCredentialFieldValue reports whether value is a credential token or
// header value that every client sending it transmits unchanged as one
// HTTP/1.1 field value (RCF-R113, RCF-R114): one or more visible ASCII
// characters (0x21 through 0x7E), with runs of space or horizontal tab only
// between them. It is the only Go definition of that grammar; the Runtime
// credential constructors, stored material decoding, LLM credential tokens and
// RuntimeSettings validation apply it, and callers apply the byte bound of
// their field.
//
// The grammar is the intersection of what the senders accept. The Runtime's
// httpx encodes a header value as ASCII and h11 refuses NUL, CR, LF, VT, FF
// and leading or trailing space or tab (HTP-R33); the Server's net/http
// refuses every other control character and trims surrounding whitespace.
// api/testdata/v1alpha1/credential-field-value-cases.json holds Go, the Python
// Runtime, the private allocation schema and the public OpenAPI to it.
func ValidCredentialFieldValue(value string) bool {
	if value == "" || !visibleASCII(value[0]) || !visibleASCII(value[len(value)-1]) {
		return false
	}
	for index := 1; index < len(value)-1; index++ {
		if character := value[index]; !visibleASCII(character) && character != ' ' && character != '\t' {
			return false
		}
	}
	return true
}

func visibleASCII(character byte) bool { return character >= 0x21 && character <= 0x7e }
