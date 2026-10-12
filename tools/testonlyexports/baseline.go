package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// baselineEntry is one exported identifier that lint tolerates without a
// production reference until its task resolves it.
type baselineEntry struct {
	key  string
	task string
	line int
}

var taskIDPattern = regexp.MustCompile(`^[A-Z][A-Z0-9]*-[0-9]+$`)

// parseBaseline reads lines of "<identifier> <task ID>". Blank lines and
// lines that start with # are ignored.
func parseBaseline(source string) ([]baselineEntry, []string) {
	var entries []baselineEntry
	var problems []string
	seen := map[string]int{}
	for index, line := range strings.Split(source, "\n") {
		number := index + 1
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) != 2 || !taskIDPattern.MatchString(fields[1]) {
			problems = append(problems, fmt.Sprintf("%d: a baseline line is: <identifier> <task ID>", number))
			continue
		}
		if previous, ok := seen[fields[0]]; ok {
			problems = append(problems, fmt.Sprintf("%d: %s is already listed at line %d", number, fields[0], previous))
			continue
		}
		seen[fields[0]] = number
		entries = append(entries, baselineEntry{key: fields[0], task: fields[1], line: number})
	}
	return entries, problems
}

// openTaskIn returns a function that describes the state of a task under
// root/tasks: "" for an open task (a file in tasks/), otherwise why the task
// cannot carry a baseline entry.
func openTaskIn(root string) func(string) string {
	return func(task string) string {
		name := strings.ToLower(task) + "-*.yml"
		if open, _ := filepath.Glob(filepath.Join(root, "tasks", name)); len(open) == 1 {
			return ""
		}
		if done, _ := filepath.Glob(filepath.Join(root, "tasks", "done", name)); len(done) == 1 {
			return "is finished; resolve the identifier or name the open task that does"
		}
		return "has no task file; name the open task that resolves the identifier"
	}
}

// check compares the findings of result with the baseline and returns one
// message per problem, sorted: a production import of a test helper package,
// which the baseline cannot list, a finding the baseline does not list, an
// entry that matches no finding, and an entry whose task is not open.
func check(result Result, entries []baselineEntry, baselineName string, taskState func(string) string) []string {
	var problems []string
	for _, helperImport := range result.HelperImports {
		problems = append(problems, fmt.Sprintf("%s imports the test helper package %s; production code must not "+
			"depend on test code", helperImport.Importer, helperImport.Helper))
	}
	findings := result.Findings
	listed := map[string]baselineEntry{}
	for _, entry := range entries {
		listed[entry.key] = entry
	}
	found := map[string]bool{}
	for _, finding := range findings {
		found[finding.Key] = true
		if _, ok := listed[finding.Key]; ok {
			continue
		}
		problems = append(problems, fmt.Sprintf("%s:%d:%d: %s %s is exported, but %s; unexport or delete it, move it into "+
			"an export_test.go or a test helper package (a package named *test), or give it a production caller",
			finding.Position.Filename, finding.Position.Line, finding.Position.Column,
			finding.Kind, finding.Key, describeReferences(finding)))
	}
	for _, entry := range entries {
		if !found[entry.key] {
			problems = append(problems, fmt.Sprintf("%s:%d: %s now has a production reference or no longer exists; delete the entry",
				baselineName, entry.line, entry.key))
		}
		if state := taskState(entry.task); state != "" {
			problems = append(problems, fmt.Sprintf("%s:%d: %s names %s, which %s",
				baselineName, entry.line, entry.key, entry.task, state))
		}
	}
	slices.Sort(problems)
	return problems
}

func describeReferences(finding Finding) string {
	if finding.TestRefs == 0 {
		return "nothing references it"
	}
	files := finding.TestFiles
	const shown = 3
	more := ""
	if len(files) > shown {
		more = fmt.Sprintf(" and %d more files", len(files)-shown)
		files = files[:shown]
	}
	references := "references"
	if finding.TestRefs == 1 {
		references = "reference"
	}
	return fmt.Sprintf("only tests reference it (%d %s in %s%s)",
		finding.TestRefs, references, strings.Join(files, ", "), more)
}
