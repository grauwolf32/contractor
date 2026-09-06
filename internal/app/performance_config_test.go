package app

import (
	"strings"
	"testing"
)

func TestPerformanceConfiguration(t *testing.T) {
	for _, metrics := range []string{"true", "false"} {
		for _, profiling := range []string{"true", "false"} {
			cfg, err := ParseConfig([]string{"--performance-metrics=" + metrics, "--pprof=" + profiling}, func(string) string { return "" })
			if err != nil || cfg.PerformanceMetrics != (metrics == "true") || cfg.Pprof != (profiling == "true") || cfg.PprofListen != "127.0.0.1:6060" {
				t.Fatalf("combination %s/%s: %+v, %v", metrics, profiling, cfg, err)
			}
		}
	}
	cfg, err := ParseConfig(nil, func(string) string { return "" })
	if err != nil || !cfg.PerformanceMetrics || cfg.Pprof {
		t.Fatalf("defaults: %+v %v", cfg, err)
	}
	env := map[string]string{"CONTRACTOR_PERFORMANCE_METRICS": "false", "CONTRACTOR_PPROF": "true", "CONTRACTOR_PPROF_LISTEN": "[::1]:6061"}
	cfg, err = ParseConfig(nil, func(key string) string { return env[key] })
	if err != nil || cfg.PerformanceMetrics || !cfg.Pprof || cfg.PprofListen != "[::1]:6061" {
		t.Fatalf("env: %+v %v", cfg, err)
	}
	for key := range env {
		env[key] = "secret-invalid"
	}
	cfg, err = ParseConfig([]string{"--performance-metrics", "--pprof=false", "--pprof-listen=127.0.0.2:65535"}, func(key string) string { return env[key] })
	if err != nil || !cfg.PerformanceMetrics || cfg.Pprof {
		t.Fatalf("CLI override: %+v %v", cfg, err)
	}
}

func TestPerformanceConfigurationRejectsInvalidEffectiveValues(t *testing.T) {
	for _, key := range []string{"CONTRACTOR_PERFORMANCE_METRICS", "CONTRACTOR_PPROF", "CONTRACTOR_PPROF_LISTEN"} {
		_, err := ParseConfig(nil, func(name string) string {
			if name == key {
				return "secret-invalid"
			}
			return ""
		})
		if err == nil || strings.Contains(err.Error(), "secret-invalid") {
			t.Fatalf("unsafe/missing error for %s: %v", key, err)
		}
	}
	for _, flag := range []string{"--pprof=1", "--pprof=", "--performance-metrics=TRUE", "--performance-metrics=secret-invalid"} {
		_, err := ParseConfig([]string{flag}, func(string) string { return "" })
		if err == nil || strings.Contains(err.Error(), "secret-invalid") {
			t.Fatalf("unsafe/missing error for boolean: %v", err)
		}
	}
	for _, address := range []string{"localhost:6060", "0.0.0.0:6060", "[::]:6060", "192.0.2.1:6060", "127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:http", "127.0.0.1:+80", "[::1%lo]:6060", "127.0.0.1", ""} {
		_, err := ParseConfig([]string{"--pprof=false", "--pprof-listen=" + address}, func(string) string { return "" })
		if err == nil {
			t.Fatalf("accepted invalid address %q even with profiling disabled", address)
		}
	}
}
