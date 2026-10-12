// Command testonlyexports fails make lint when an exported identifier of the
// module has no production reference: staticcheck's U1000 reports only
// unexported code, so exported surface that only tests use would otherwise
// grow unnoticed (AGENTS.md 3.7).
//
// It loads every package with its tests, including the test files behind the
// -tags build tags, and reports each exported package-level function, type,
// constant and variable, each exported method of an exported type and each
// exported method of an exported interface that no production code refers
// to, whether only tests use it or nothing does. Production code is every
// file outside test code; generated files and the commands under cmd/ and
// tools/ are production code. A reference from inside the identifier's own
// declaration, such as a recursive call or a method's receiver, and a
// compile-time interface check, var _ I = (*T)(nil), are not uses. A method
// also counts as used when it implements a used interface method (see
// Analyze). Struct fields are not checked: encoding/json and database scans
// use them without a reference.
//
// The allowlist rule: a package that exists for tests, which by its name
// ends in "test" (such as internal/artifacts/artifactstest), is test code as
// a whole, like _test.go files and the process tests under tests/. Its
// exported helpers are never reported, and its references count as test
// references. A test hook of one package belongs in that package's
// export_test.go; a helper that other packages' tests share belongs in a
// test helper package. Production code must not import a test helper
// package: the check fails on such an import, and no baseline entry can
// excuse it, so the rule cannot hide a production use.
//
// The baseline file lists identifiers that lint tolerates for now, one
// "<identifier> <task ID>" line each, where the task is an open task file in
// tasks/ that resolves the identifier. The check fails on an identifier that
// the baseline does not list, on an entry that no longer matches one, and on
// an entry whose task is finished or missing. Fixing one identifier can
// expose another that only it used; the next run reports that one.
//
// Usage:
//
//	go -C tools/testonlyexports run . [-tags integration,e2e] [-baseline FILE] MODULE_DIR
//
// FILE is relative to MODULE_DIR unless it is absolute. The exit status is 1
// when the check fails and 2 when the module cannot be loaded.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("testonlyexports", flag.ContinueOnError)
	flags.SetOutput(stderr)
	tags := flags.String("tags", "integration,e2e", "comma-separated build tags that select further test files")
	baseline := flags.String("baseline", "", "file of tolerated identifiers, relative to the module directory")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: testonlyexports [-tags TAGS] [-baseline FILE] MODULE_DIR")
		return 2
	}
	root := flags.Arg(0)
	var entries []baselineEntry
	if *baseline != "" {
		path := *baseline
		if !filepath.IsAbs(path) {
			path = filepath.Join(root, path)
		}
		source, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		var problems []string
		entries, problems = parseBaseline(string(source))
		if len(problems) > 0 {
			for _, problem := range problems {
				fmt.Fprintf(stderr, "%s:%s\n", *baseline, problem)
			}
			return 1
		}
	}
	var buildTags []string
	if *tags != "" {
		buildTags = strings.Split(*tags, ",")
	}
	result, err := Analyze(Options{Dir: root, Tags: buildTags})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	problems := check(result, entries, *baseline, openTaskIn(root))
	for _, problem := range problems {
		fmt.Fprintln(stdout, problem)
	}
	if len(problems) > 0 {
		noun := "problems"
		if len(problems) == 1 {
			noun = "problem"
		}
		fmt.Fprintf(stderr, "testonlyexports: %d %s; the rules are in tools/testonlyexports/main.go\n", len(problems), noun)
		return 1
	}
	return 0
}
