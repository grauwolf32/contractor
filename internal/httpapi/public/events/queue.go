package events

import "sync"

const (
	maximumQueuedFrames = 256
	maximumQueuedBytes  = 1024 * 1024
)

type outboundQueue struct {
	mu     sync.Mutex
	frames chan []byte
	bytes  int
	closed bool
}

func newOutboundQueue() *outboundQueue {
	return &outboundQueue{frames: make(chan []byte, maximumQueuedFrames)}
}

func (q *outboundQueue) enqueue(frame []byte) bool {
	if len(frame) == 0 || len(frame) > maximumServerFrameBytes {
		return false
	}
	copyOfFrame := append([]byte(nil), frame...)
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed || len(q.frames) >= maximumQueuedFrames || q.bytes+len(copyOfFrame) > maximumQueuedBytes {
		return false
	}
	q.bytes += len(copyOfFrame)
	q.frames <- copyOfFrame
	return true
}

func (q *outboundQueue) channel() <-chan []byte { return q.frames }

func (q *outboundQueue) consumed(frame []byte) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.bytes -= len(frame)
	if q.bytes < 0 {
		q.bytes = 0
	}
}

func (q *outboundQueue) close() {
	q.mu.Lock()
	q.closed = true
	q.mu.Unlock()
}

func (q *outboundQueue) size() (int, int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.frames), q.bytes
}
