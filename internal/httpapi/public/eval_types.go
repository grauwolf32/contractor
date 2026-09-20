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
	PutRecord(context.Context, evalstore.Scope, string, string, evaldomain.Frozen, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	Select(context.Context, evalstore.Scope, string, evaldomain.SelectionInput, evaldomain.MutationIdentity) (evalstore.Receipt, error)
	Review(context.Context, string, string, string, string) (evalservice.ReviewContext, error)
	Pairs(context.Context, evalservice.PairPageParams) (evalservice.PairPage, error)
	Pair(context.Context, evalservice.MemberPageParams, string) (evalservice.PairDetail, error)
	Chart(context.Context, evalservice.ChartParams) (evalservice.ChartView, error)
	Report(context.Context, evalservice.MemberPageParams) (evalservice.Report, error)
	Executions(context.Context, string, string, string, string, int, *int64) (evalstore.InventoryPage, error)
}

type evalPageInfo struct {
	HasMore    bool    `json:"hasMore"`
	NextCursor *string `json:"nextCursor"`
}
