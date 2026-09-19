package public

import (
	"context"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

type EvalManagement interface {
	Get(context.Context, string, string) (evalservice.ExperimentView, error)
	List(context.Context, evalstore.SummaryPageParams) (evalstore.SummaryPage, error)
	Datasets(context.Context, evalstore.Scope, string, string, int, *int64) (evalstore.DatasetPage, error)
	Dataset(context.Context, evalstore.Scope, string, string) (evalstore.DatasetRevision, error)
	PutDataset(context.Context, evalstore.Scope, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	Create(context.Context, evalstore.Scope, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	ScopeForMutation(context.Context, string, string, string, string) (evalstore.Scope, error)
	UpdateDraft(context.Context, evalstore.Scope, string, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	Command(context.Context, evalstore.Scope, string, evaldomain.Command, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	Submit(context.Context, evalstore.Scope, string, string, evaldomain.Submission, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	Delete(context.Context, evalstore.Scope, string, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	GetCommand(context.Context, string, string, string) (evalstore.CommandRecord, *string, error)
	Members(context.Context, evalservice.MemberPageParams) (evalservice.MemberPage, error)
}

type evalPageInfo struct {
	HasMore    bool    `json:"hasMore"`
	NextCursor *string `json:"nextCursor"`
}
