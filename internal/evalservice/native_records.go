package evalservice

import (
	"context"
	"encoding/json"
	"reflect"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func (s *Service) collectNativeRecords(ctx context.Context, st *evalstore.Store, db pg.DBTX, e evalstore.Experiment, m evalstore.Member, claim evalstore.Claim, checks []evaldomain.Check, result evaldomain.ResultInput) error {
	prior, err := st.LatestNativeRecord(ctx, e.OwnerID, e.ID, m.MemberID, evaldomain.RecordKindResult)
	if err != nil {
		return err
	}
	if prior != nil {
		var old evaldomain.ResultInput
		if err = json.Unmarshal(prior.Document.Bytes(), &old); err != nil {
			return err
		}
		old.PreviousResultSHA256 = nil
		if reflect.DeepEqual(old, result) {
			return nil
		}
		result.PreviousResultSHA256 = &prior.SHA256
	}
	raw, err := jsonBytes(result)
	if err != nil {
		return err
	}
	doc, err := evaldomain.Freeze("ResultInput", raw)
	if err != nil {
		return err
	}
	if _, err = st.SaveNativeRecord(ctx, scope(e), e.ID, m.MemberID, claim, doc); err != nil {
		return err
	}
	checkResults, err := runNativeChecks(ctx, db, checks, m.Recipe.Case.Outputs, result)
	if err != nil {
		return err
	}
	assessment := evaldomain.AssessmentInput{SchemaVersion: evaldomain.AssessmentSchemaVersion, Source: evaldomain.AssessmentSource{Kind: "native"}, ResultSHA256: doc.Digest(), Checks: checkResults}
	previousAssessment, err := st.LatestNativeRecord(ctx, e.OwnerID, e.ID, m.MemberID, evaldomain.RecordKindAssessment)
	if err != nil {
		return err
	}
	if previousAssessment != nil {
		assessment.PreviousAssessmentSHA256 = &previousAssessment.SHA256
	}
	raw, err = jsonBytes(assessment)
	if err != nil {
		return err
	}
	assessmentDoc, err := evaldomain.Freeze("AssessmentInput", raw)
	if err != nil {
		return err
	}
	if _, err = st.SaveNativeRecord(ctx, scope(e), e.ID, m.MemberID, claim, assessmentDoc); err != nil {
		return err
	}
	return st.SelectFirstNative(ctx, scope(e), e.ID, m.MemberID, claim, doc, assessmentDoc)
}
