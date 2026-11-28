package runtimewire

const (
	BulkChunkBytes  = 256 << 10
	BulkHeaderBytes = 1 << 20
	BulkFrameBytes  = BulkHeaderBytes + 1024
	BulkWindowBytes = 1 << 20
)

// BulkChunkSize is the only admitted size of the next payload chunk. Exact
// lengths make the declared aggregate bound a frame/work bound as well.
func BulkChunkSize(remaining int) int { return min(BulkChunkBytes, remaining) }
