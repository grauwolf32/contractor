package contracts

const (
	// MaxGitImportURLBytes bounds the caller's original UTF-8 input (LIM-GIT-23).
	MaxGitImportURLBytes = 4096
	// MaxGitSourceURLBytes leaves room for percent encoding every input byte
	// in immutable canonical provenance (LIM-GIT-24). The supported URL forms'
	// unexpanded structural bytes also cover inserted user/port/SCP defaults.
	MaxGitSourceURLBytes = 3 * MaxGitImportURLBytes
)
