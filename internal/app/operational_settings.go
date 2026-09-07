package app

import (
	"flag"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// OperationalSettings contains resolved process budgets, independent of transport
// request timeouts and persisted Workflow/Allocation deadlines.
type OperationalSettings struct {
	Scheduler            SchedulerSettings
	RuntimeLifecycle     RuntimeLifecycleSettings
	ProjectLifecycle     ProjectLifecycleSettings
	AuditController      AuditControllerSettings
	Database             DatabaseSettings
	A2A                  A2ASettings
	CredentialManagement CredentialManagementSettings
}

type SchedulerSettings struct {
	OperationTimeout    time.Duration
	FinalizationTimeout time.Duration
	AbortTimeout        time.Duration
}

type RuntimeLifecycleSettings struct {
	CleanupTimeout time.Duration
}

type ProjectLifecycleSettings struct {
	OperationTimeout time.Duration
	ClaimDuration    time.Duration
}

type AuditControllerSettings struct {
	PollInterval     time.Duration
	ClaimLease       time.Duration
	OperationTimeout time.Duration
	ClaimBatch       int
}

type DatabaseSettings struct {
	ConnectTimeout         time.Duration
	AcquireTimeout         time.Duration
	QueryTimeout           time.Duration
	StatementTimeout       time.Duration
	LockTimeout            time.Duration
	IdleTransactionTimeout time.Duration
}

type A2ASettings struct {
	PollInterval time.Duration
}

type CredentialManagementSettings struct {
	ConnectTimeout time.Duration
	RequestTimeout time.Duration
}

type operationalSpec struct {
	Scheduler struct {
		OperationTimeout    *string `yaml:"operationTimeout"`
		FinalizationTimeout *string `yaml:"finalizationTimeout"`
		AbortTimeout        *string `yaml:"abortTimeout"`
	} `yaml:"scheduler"`
	RuntimeLifecycle struct {
		CleanupTimeout *string `yaml:"cleanupTimeout"`
	} `yaml:"runtimeLifecycle"`
	ProjectLifecycle struct {
		OperationTimeout *string `yaml:"operationTimeout"`
		ClaimDuration    *string `yaml:"claimDuration"`
	} `yaml:"projectLifecycle"`
	AuditController struct {
		PollInterval     *string `yaml:"pollInterval"`
		ClaimLease       *string `yaml:"claimLease"`
		OperationTimeout *string `yaml:"operationTimeout"`
		ClaimBatch       *int    `yaml:"claimBatch"`
	} `yaml:"auditController"`
	Database struct {
		ConnectTimeout         *string `yaml:"connectTimeout"`
		AcquireTimeout         *string `yaml:"acquireTimeout"`
		QueryTimeout           *string `yaml:"queryTimeout"`
		StatementTimeout       *string `yaml:"statementTimeout"`
		LockTimeout            *string `yaml:"lockTimeout"`
		IdleTransactionTimeout *string `yaml:"idleTransactionTimeout"`
	} `yaml:"database"`
	A2A struct {
		PollInterval *string `yaml:"pollInterval"`
	} `yaml:"a2a"`
	CredentialManagement struct {
		ConnectTimeout *string `yaml:"connectTimeout"`
		RequestTimeout *string `yaml:"requestTimeout"`
	} `yaml:"credentialManagement"`
}

func defaultOperationalSettings() OperationalSettings {
	return OperationalSettings{
		Scheduler: SchedulerSettings{
			OperationTimeout:    30 * time.Second,
			FinalizationTimeout: 10 * time.Second,
			AbortTimeout:        10 * time.Second,
		},
		RuntimeLifecycle: RuntimeLifecycleSettings{
			CleanupTimeout: 30 * time.Second,
		},
		ProjectLifecycle: ProjectLifecycleSettings{
			OperationTimeout: 30 * time.Second,
			ClaimDuration:    time.Minute,
		},
		AuditController: AuditControllerSettings{
			PollInterval:     time.Second,
			ClaimLease:       30 * time.Second,
			OperationTimeout: 10 * time.Second,
			ClaimBatch:       8,
		},
		Database: DatabaseSettings{
			ConnectTimeout:         5 * time.Second,
			AcquireTimeout:         2 * time.Second,
			QueryTimeout:           20 * time.Second,
			StatementTimeout:       15 * time.Second,
			LockTimeout:            2 * time.Second,
			IdleTransactionTimeout: 30 * time.Second,
		},
		A2A: A2ASettings{
			PollInterval: 100 * time.Millisecond,
		},
		CredentialManagement: CredentialManagementSettings{
			ConnectTimeout: 3 * time.Second,
			RequestTimeout: 15 * time.Second,
		},
	}
}

type operationalDuration struct {
	path   string
	flag   string
	target *time.Duration
	source *string
}

func (s *OperationalSettings) durations(spec operationalSpec) []operationalDuration {
	return []operationalDuration{
		{"scheduler.operationTimeout", "scheduler-operation-timeout", &s.Scheduler.OperationTimeout, spec.Scheduler.OperationTimeout},
		{"scheduler.finalizationTimeout", "scheduler-finalization-timeout", &s.Scheduler.FinalizationTimeout, spec.Scheduler.FinalizationTimeout},
		{"scheduler.abortTimeout", "scheduler-abort-timeout", &s.Scheduler.AbortTimeout, spec.Scheduler.AbortTimeout},
		{"runtimeLifecycle.cleanupTimeout", "runtime-lifecycle-cleanup-timeout", &s.RuntimeLifecycle.CleanupTimeout, spec.RuntimeLifecycle.CleanupTimeout},
		{"projectLifecycle.operationTimeout", "project-lifecycle-operation-timeout", &s.ProjectLifecycle.OperationTimeout, spec.ProjectLifecycle.OperationTimeout},
		{"projectLifecycle.claimDuration", "project-lifecycle-claim-duration", &s.ProjectLifecycle.ClaimDuration, spec.ProjectLifecycle.ClaimDuration},
		{"auditController.pollInterval", "audit-controller-poll-interval", &s.AuditController.PollInterval, spec.AuditController.PollInterval},
		{"auditController.claimLease", "audit-controller-claim-lease", &s.AuditController.ClaimLease, spec.AuditController.ClaimLease},
		{"auditController.operationTimeout", "audit-controller-operation-timeout", &s.AuditController.OperationTimeout, spec.AuditController.OperationTimeout},
		{"database.connectTimeout", "database-connect-timeout", &s.Database.ConnectTimeout, spec.Database.ConnectTimeout},
		{"database.acquireTimeout", "database-acquire-timeout", &s.Database.AcquireTimeout, spec.Database.AcquireTimeout},
		{"database.queryTimeout", "database-query-timeout", &s.Database.QueryTimeout, spec.Database.QueryTimeout},
		{"database.statementTimeout", "database-statement-timeout", &s.Database.StatementTimeout, spec.Database.StatementTimeout},
		{"database.lockTimeout", "database-lock-timeout", &s.Database.LockTimeout, spec.Database.LockTimeout},
		{"database.idleTransactionTimeout", "database-idle-transaction-timeout", &s.Database.IdleTransactionTimeout, spec.Database.IdleTransactionTimeout},
		{"a2a.pollInterval", "a2a-poll-interval", &s.A2A.PollInterval, spec.A2A.PollInterval},
		{"credentialManagement.connectTimeout", "credential-management-connect-timeout", &s.CredentialManagement.ConnectTimeout, spec.CredentialManagement.ConnectTimeout},
		{"credentialManagement.requestTimeout", "credential-management-request-timeout", &s.CredentialManagement.RequestTimeout, spec.CredentialManagement.RequestTimeout},
	}
}

func (s *OperationalSettings) applySpec(spec operationalSpec) error {
	for _, setting := range s.durations(spec) {
		if err := setDuration(setting.target, setting.source, setting.path); err != nil {
			return err
		}
	}
	if spec.AuditController.ClaimBatch != nil {
		s.AuditController.ClaimBatch = *spec.AuditController.ClaimBatch
	}
	return nil
}

func (s *OperationalSettings) registerFlags(flags *flag.FlagSet, getenv func(string) string) error {
	for _, setting := range s.durations(operationalSpec{}) {
		env := "CONTRACTOR_" + strings.ToUpper(strings.ReplaceAll(setting.flag, "-", "_"))
		if encoded := getenv(env); encoded != "" {
			parsed, err := time.ParseDuration(encoded)
			if err != nil {
				return fmt.Errorf("%s must be a duration", env)
			}
			*setting.target = parsed
		}
		flags.DurationVar(setting.target, setting.flag, *setting.target, "ServerConfig spec."+setting.path)
	}
	if encoded := getenv("CONTRACTOR_AUDIT_CONTROLLER_CLAIM_BATCH"); encoded != "" {
		parsed, err := strconv.Atoi(encoded)
		if err != nil {
			return fmt.Errorf("CONTRACTOR_AUDIT_CONTROLLER_CLAIM_BATCH must be an integer")
		}
		s.AuditController.ClaimBatch = parsed
	}
	flags.IntVar(&s.AuditController.ClaimBatch, "audit-controller-claim-batch", s.AuditController.ClaimBatch, "maximum concurrently claimed Audits")
	return nil
}

func (s *OperationalSettings) validate() error {
	for _, setting := range s.durations(operationalSpec{}) {
		if *setting.target <= 0 {
			return fmt.Errorf("spec.%s must be positive", setting.path)
		}
	}
	audit := s.AuditController
	// Division avoids overflow for a large, syntactically valid duration.
	if audit.ClaimLease < time.Second || audit.ClaimLease > 5*time.Minute || audit.OperationTimeout > audit.ClaimLease/2 {
		return fmt.Errorf("spec.auditController.claimLease must be 1s..5m and at least twice operationTimeout")
	}
	if audit.ClaimBatch < 1 || audit.ClaimBatch > auditstore.MaxClaimBatch {
		return fmt.Errorf("spec.auditController.claimBatch must be 1..%d", auditstore.MaxClaimBatch)
	}
	if s.ProjectLifecycle.OperationTimeout >= s.ProjectLifecycle.ClaimDuration {
		return fmt.Errorf("spec.projectLifecycle.operationTimeout must be less than claimDuration")
	}
	credentials := s.CredentialManagement
	if credentials.ConnectTimeout > time.Minute || credentials.RequestTimeout > 2*time.Minute {
		return fmt.Errorf("spec.credentialManagement requires connectTimeout <= 1m and requestTimeout <= 2m")
	}
	return s.Database.budgets().Validate()
}

func (s DatabaseSettings) budgets() persistencepostgres.Budgets {
	return persistencepostgres.Budgets{
		AcquireTimeout: s.AcquireTimeout, QueryTimeout: s.QueryTimeout,
		StatementTimeout: s.StatementTimeout, LockTimeout: s.LockTimeout,
		IdleTransactionTimeout: s.IdleTransactionTimeout,
	}
}

func (s DatabaseSettings) poolOptions(logger *slog.Logger) persistencepostgres.PoolOptions {
	return persistencepostgres.PoolOptions{ConnectTimeout: s.ConnectTimeout, Budgets: s.budgets(), Logger: logger, RejectTimeoutOverrides: true}
}

// logEffective emits only non-secret resolved operational values.
func (s OperationalSettings) logEffective(logger *slog.Logger, requestTimeout time.Duration) {
	attrs := []any{"runtimeRequestTimeout", requestTimeout.String()}
	for _, setting := range s.durations(operationalSpec{}) {
		attrs = append(attrs, setting.path, setting.target.String())
	}
	attrs = append(attrs, "auditController.claimBatch", s.AuditController.ClaimBatch)
	logger.Info("effective operational settings", attrs...)
}
