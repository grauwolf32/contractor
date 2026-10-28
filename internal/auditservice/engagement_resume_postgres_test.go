package auditservice

import (
	"bytes"
	"testing"
	"time"
)

func TestEngagementResumePreservesFiniteAllowanceAndReplay(t *testing.T) {
	for _, test := range []struct {
		name      string
		override  *int
		remainder time.Duration
		want      time.Duration
	}{
		{name: "zero", override: new(0), want: 120 * time.Second},
		{name: "oversized-positive", override: new(600), want: 120 * time.Second},
		{name: "smaller-positive", override: new(30), want: 30 * time.Second},
		{name: "omitted-exhausted", want: 120 * time.Second},
		{name: "omitted-remainder", remainder: 40 * time.Second, want: 40 * time.Second},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newEngagementFixture(t, 1)
			now := f.now
			f.service.now = func() time.Time { return now }
			draft, err := f.create(t, "resume-engagement", f.document("http://127.0.0.1"))
			if err != nil {
				t.Fatal(err)
			}
			started, err := f.service.Start(f.ctx, f.start(draft))
			if err != nil {
				t.Fatal(err)
			}
			paused, err := f.service.Pause(f.ctx, MutationParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: started.Audit.Revision, IdempotencyKey: "pause-engagement", RequestDigest: serviceTestDigest("pause-engagement")})
			if err != nil {
				t.Fatal(err)
			}
			// Fix the stored pause instant relative to the injected start clock. The
			// test asserts budget arithmetic without depending on elapsed wall time.
			pauseAt := started.Audit.DeadlineAt.Add(-test.remainder)
			if _, err = f.pool.Exec(f.ctx, `UPDATE audits SET paused_at=$2 WHERE audit_id=$1`, draft.AuditID, pauseAt); err != nil {
				t.Fatal(err)
			}
			now = now.Add(24 * time.Hour)
			expectedDeadline := now.Add(test.want)
			params := MutationParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: paused.Audit.Revision, IdempotencyKey: "resume-engagement", RequestDigest: serviceTestDigest("resume-engagement"), DeadlineSeconds: test.override}
			resumed, err := f.service.Resume(f.ctx, params)
			if err != nil {
				t.Fatal(err)
			}
			if resumed.Audit.DeadlineAt == nil || !resumed.Audit.DeadlineAt.Equal(expectedDeadline) {
				t.Fatalf("resumed deadline=%v want %v", resumed.Audit.DeadlineAt, expectedDeadline)
			}
			if !bytes.Equal(resumed.Audit.BaselineSnapshot, started.Audit.BaselineSnapshot) {
				t.Fatal("resume rewrote engagement authority")
			}
			now = now.Add(time.Hour)
			replayed, err := f.service.Resume(f.ctx, params)
			if err != nil || !replayed.Replayed || replayed.Audit.Revision != resumed.Audit.Revision || replayed.Audit.DeadlineAt == nil || !replayed.Audit.DeadlineAt.Equal(expectedDeadline) {
				t.Fatalf("resume replay changed bounded deadline: %+v %v", replayed, err)
			}
		})
	}
}
