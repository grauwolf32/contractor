package app

import (
	"flag"
	"fmt"
	"strconv"

	"github.com/grauwolf32/contractor/internal/gitimport"
)

type gitImportSpec struct {
	Timeout          *string `yaml:"timeout"`
	MaxReceivedBytes *int64  `yaml:"maxReceivedBytes"`
	MaxDecodedBytes  *int64  `yaml:"maxDecodedBytes"`
	MaxExpandedBytes *int64  `yaml:"maxExpandedBytes"`
	MaxFileBytes     *int64  `yaml:"maxFileBytes"`
	MaxEntries       *int    `yaml:"maxEntries"`
}

func applyGitImportSpec(limits *gitimport.Limits, spec gitImportSpec) error {
	if err := setDuration(&limits.Timeout, spec.Timeout, "gitImport.timeout"); err != nil {
		return err
	}
	for _, value := range []struct{ target, source *int64 }{
		{&limits.MaxReceivedBytes, spec.MaxReceivedBytes},
		{&limits.MaxDecodedBytes, spec.MaxDecodedBytes},
		{&limits.Source.MaxExpandedBytes, spec.MaxExpandedBytes},
		{&limits.Source.MaxFileBytes, spec.MaxFileBytes},
	} {
		if value.source != nil {
			*value.target = *value.source
		}
	}
	if spec.MaxEntries != nil {
		limits.Source.MaxEntries = *spec.MaxEntries
	}
	return nil
}

func gitImportEnvironment(limits *gitimport.Limits, getenv func(string) string) error {
	if value := getenv("CONTRACTOR_GIT_IMPORT_TIMEOUT"); value != "" {
		if err := setDuration(&limits.Timeout, &value, "gitImport.timeout"); err != nil {
			return err
		}
	}
	for _, value := range []struct {
		name   string
		target *int64
	}{
		{"CONTRACTOR_GIT_IMPORT_MAX_RECEIVED_BYTES", &limits.MaxReceivedBytes},
		{"CONTRACTOR_GIT_IMPORT_MAX_DECODED_BYTES", &limits.MaxDecodedBytes},
		{"CONTRACTOR_GIT_IMPORT_MAX_EXPANDED_BYTES", &limits.Source.MaxExpandedBytes},
		{"CONTRACTOR_GIT_IMPORT_MAX_FILE_BYTES", &limits.Source.MaxFileBytes},
	} {
		if raw := getenv(value.name); raw != "" {
			parsed, err := strconv.ParseInt(raw, 10, 64)
			if err != nil {
				return fmt.Errorf("%s must be an integer byte count", value.name)
			}
			*value.target = parsed
		}
	}
	if raw := getenv("CONTRACTOR_GIT_IMPORT_MAX_ENTRIES"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			return fmt.Errorf("CONTRACTOR_GIT_IMPORT_MAX_ENTRIES must be an integer")
		}
		limits.Source.MaxEntries = parsed
	}
	return nil
}

func gitImportFlags(flags *flag.FlagSet, limits *gitimport.Limits) {
	flags.DurationVar(&limits.Timeout, "git-import-timeout", limits.Timeout, "whole Git import timeout (maximum 30m)")
	flags.Int64Var(&limits.MaxReceivedBytes, "git-import-max-received-bytes", limits.MaxReceivedBytes, "cumulative Git wire byte budget (maximum 1 GiB)")
	flags.Int64Var(&limits.MaxDecodedBytes, "git-import-max-decoded-bytes", limits.MaxDecodedBytes, "cumulative decoded Git object byte budget (maximum 2 GiB)")
	flags.Int64Var(&limits.Source.MaxExpandedBytes, "git-import-max-expanded-bytes", limits.Source.MaxExpandedBytes, "expanded source archive byte budget (maximum 1 GiB)")
	flags.Int64Var(&limits.Source.MaxFileBytes, "git-import-max-file-bytes", limits.Source.MaxFileBytes, "one source file byte budget (maximum 64 MiB)")
	flags.IntVar(&limits.Source.MaxEntries, "git-import-max-entries", limits.Source.MaxEntries, "source tree entry budget, including directories (maximum 50000)")
}
