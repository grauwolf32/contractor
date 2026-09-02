package controlplane

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestConfirmedLeaseRequiresEchoOfIssuedAck(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registration := testRegistration("agent-1")
	if _, err := registry.Register(registration); err != nil {
		t.Fatal(err)
	}

	first := heartbeat("agent-1", 1, 0)
	response1, err := registry.Heartbeat(first)
	if err != nil || response1.AckSeq != 1 {
		t.Fatalf("first heartbeat = (%+v, %v)", response1, err)
	}
	afterFirst, _ := registry.GetAgent("agent-1")
	if !afterFirst.ConfirmedLeaseExpiresAt.IsZero() {
		t.Fatalf("unconfirmed first heartbeat established lease %s", afterFirst.ConfirmedLeaseExpiresAt)
	}

	clock.Advance(5 * time.Second)
	if _, err := registry.Heartbeat(heartbeat("agent-1", 2, 1)); err != nil {
		t.Fatal(err)
	}
	confirmed, _ := registry.GetAgent("agent-1")
	wantExpiry := clock.Now().Add(time.Minute)
	if confirmed.LastConfirmedAckSeq != 1 || !confirmed.ConfirmedLeaseExpiresAt.Equal(wantExpiry) {
		t.Fatalf("confirmed lease = ack %d, expiry %s; want ack 1, expiry %s", confirmed.LastConfirmedAckSeq, confirmed.ConfirmedLeaseExpiresAt, wantExpiry)
	}

	clock.Advance(5 * time.Second)
	replayed, err := registry.Heartbeat(heartbeat("agent-1", 2, 1))
	if err != nil || replayed.AckSeq != 2 {
		t.Fatalf("replayed heartbeat = (%+v, %v)", replayed, err)
	}
	if _, err := registry.Heartbeat(heartbeat("agent-1", 1, 0)); err != nil {
		t.Fatalf("cached out-of-order heartbeat: %v", err)
	}
	if _, err := registry.Heartbeat(heartbeat("agent-1", 3, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(heartbeat("agent-1", 4, 999)); err != nil {
		t.Fatal(err)
	}
	unchanged, _ := registry.GetAgent("agent-1")
	if !unchanged.ConfirmedLeaseExpiresAt.Equal(wantExpiry) || unchanged.LastConfirmedAckSeq != 1 {
		t.Fatalf("old/unknown ack advanced lease: %+v", unchanged)
	}

	clock.Advance(5 * time.Second)
	if _, err := registry.Heartbeat(heartbeat("agent-1", 5, 3)); err != nil {
		t.Fatal(err)
	}
	advanced, _ := registry.GetAgent("agent-1")
	if advanced.LastConfirmedAckSeq != 3 || !advanced.ConfirmedLeaseExpiresAt.Equal(clock.Now().Add(time.Minute)) {
		t.Fatalf("new echoed ack did not advance lease: %+v", advanced)
	}
}

func TestReserveAllIsAtomicWithInsufficientCapacity(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	template := testTemplate(t)
	_, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-two",
		Bindings: []BindingRequirement{
			testBinding(t, "first", "first", template),
			testBinding(t, "second", "second", template),
		},
	})
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("two-slot reservation error = %v", err)
	}
	reservation, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-one",
		Bindings: []BindingRequirement{testBinding(t, "only", "only", template)},
	})
	if err != nil || len(reservation) != 1 || reservation[0].Grant.RuntimeInstanceID != "agent-1" {
		t.Fatalf("slot was mutated by failed reservation: (%+v, %v)", reservation, err)
	}
}

func TestReserveAllFindsCompleteSpecialistGeneralistAssignment(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	specialist := testRegistration("agent-a-specialist")
	specialist.SupportedToolsets = append(specialist.SupportedToolsets, contracts.ToolsetCapability{
		Ref: "likec4@1", Tools: []string{"validate_likec4"},
	})
	registerReadyWith(t, registry, specialist)
	registerReady(t, registry, "agent-b-generalist")

	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-heterogeneous", StageExecutionID: "stage-heterogeneous",
		Bindings: []BindingRequirement{
			testBinding(t, "a-generic", "generic", testTemplate(t)),
			testBinding(t, "b-validator", "validator", likeC4ValidationTemplate(t)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(reservations) != 2 ||
		reservations[0].Grant.LogicalAgentName != "a-generic" ||
		reservations[0].Grant.RuntimeInstanceID != "agent-b-generalist" ||
		reservations[1].Grant.LogicalAgentName != "b-validator" ||
		reservations[1].Grant.RuntimeInstanceID != "agent-a-specialist" {
		t.Fatalf("complete heterogeneous assignment = %+v", reservations)
	}
}

func TestReserveAllContainsCodeAnalysisOperationSubsets(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	shallow := testRegistration("agent-a-shallow")
	shallow.SupportedToolsets = append(shallow.SupportedToolsets, contracts.ToolsetCapability{
		Ref: "code-analysis@1", Tools: []string{"list_symbols", "search_def"},
	})
	graph := testRegistration("agent-b-graph")
	graph.SupportedToolsets = append(graph.SupportedToolsets, contracts.ToolsetCapability{
		Ref: "code-analysis@1", Tools: []string{
			"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
			"find_callees", "find_callers", "find_symbol", "functions_that_raise",
			"graph_summary", "list_symbols", "paths_between", "search_def",
		},
	})
	registerReadyWith(t, registry, shallow)
	registerReadyWith(t, registry, graph)

	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-code-analysis", StageExecutionID: "stage-code-analysis",
		Bindings: []BindingRequirement{
			testBinding(t, "a-shallow", "shallow", codeAnalysisTemplate(t, "list_symbols")),
			testBinding(t, "b-graph", "graph", codeAnalysisTemplate(t, "paths_between")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(reservations) != 2 ||
		reservations[0].Grant.RuntimeInstanceID != "agent-a-shallow" ||
		reservations[1].Grant.RuntimeInstanceID != "agent-b-graph" {
		t.Fatalf("code-analysis subset assignment = %+v", reservations)
	}
}

func TestWorkspaceModeParticipatesInCompleteCompatibility(t *testing.T) {
	t.Parallel()
	template := testTemplate(t)
	registration := testRegistrationV2("workspace-agent")
	registration.WorkspaceCapabilities = &contracts.WorkspaceCapabilitiesV2{
		Storage: contracts.WorkspaceStorageLocal,
		Modes:   []contracts.WorkspaceModeV2{contracts.WorkspaceModeDirect},
		Limits: contracts.WorkspaceLimitsV2{
			MaxFiles: 100, MaxExpandedBytes: 1024, MaxManagedTextBytes: 1024, MaxFileBytes: 1024,
		},
	}
	direct := &contracts.AllocationWorkspaceSpecV2{
		Mode: contracts.WorkspaceModeDirect,
		Sources: []contracts.AllocationWorkspaceSourceV2{{
			Artifact: exactWorkspaceRef("source", "r1"), Target: "",
		}},
	}
	overlay := contracts.CloneAllocationWorkspaceSpecV2(direct)
	overlay.Mode = contracts.WorkspaceModeOverlay
	if !isCompatible(registration, template, direct) {
		t.Fatal("direct-capable Runtime Agent rejected a direct workspace")
	}
	if isCompatible(registration, template, overlay) {
		t.Fatal("direct-only Runtime Agent accepted an overlay workspace")
	}
	if isCompatible(testRegistrationV2("no-workspace-agent"), template, direct) {
		t.Fatal("Runtime Agent without workspace capability accepted a workspace")
	}
}

func TestWorkspaceReservationReplayPinsExactProjection(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registration := testRegistrationV2("workspace-replay-agent")
	registration.WorkspaceCapabilities = &contracts.WorkspaceCapabilitiesV2{
		Storage: contracts.WorkspaceStorageLocal,
		Modes:   []contracts.WorkspaceModeV2{contracts.WorkspaceModeDirect},
		Limits: contracts.WorkspaceLimitsV2{
			MaxFiles: 100, MaxExpandedBytes: 1024,
			MaxManagedTextBytes: 1024, MaxFileBytes: 1024,
		},
	}
	principal := legacyPrincipal(registration.InstanceID)
	if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
		t.Fatal(err)
	}
	for _, beat := range []contracts.AgentHeartbeat{
		heartbeat(registration.InstanceID, 1, 0), heartbeat(registration.InstanceID, 2, 1),
	} {
		if _, err := registry.HeartbeatAuthenticated(principal.RuntimeAgentID, beat); err != nil {
			t.Fatal(err)
		}
	}
	binding := testBinding(t, "builder", "builder", testTemplate(t))
	binding.Workspace = &contracts.AllocationWorkspaceSpecV2{
		Mode: contracts.WorkspaceModeDirect,
		Sources: []contracts.AllocationWorkspaceSourceV2{{
			Artifact: exactWorkspaceRef("source", "revision-1"), Target: "",
		}},
	}
	request := ReservationRequest{
		RunID: "run-workspace", StageExecutionID: "stage-workspace",
		Bindings: []BindingRequirement{binding},
	}
	first, err := registry.ReserveAll(request)
	if err != nil || len(first) != 1 || first[0].Workspace == nil ||
		*first[0].Workspace.Sources[0].Artifact.Revision != "revision-1" {
		t.Fatalf("workspace reservation = (%+v, %v)", first, err)
	}
	first[0].Workspace.Sources[0].Target = "mutated-by-caller"
	replayed, err := registry.ReserveAll(request)
	if err != nil || replayed[0].Workspace.Sources[0].Target != "" {
		t.Fatalf("workspace replay = (%+v, %v)", replayed, err)
	}
	changed := request
	changed.Bindings = append([]BindingRequirement(nil), request.Bindings...)
	changed.Bindings[0].Workspace = contracts.CloneAllocationWorkspaceSpecV2(binding.Workspace)
	revision := "revision-2"
	changed.Bindings[0].Workspace.Sources[0].Artifact.Revision = &revision
	if _, err := registry.ReserveAll(changed); !errors.Is(err, ErrReservationConflict) {
		t.Fatalf("changed workspace replay error = %v", err)
	}
}

func exactWorkspaceRef(name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: "inputs", Name: name, Revision: &revision}
}

func TestReserveCandidateEdgesFindsCompleteNonGreedyAssignment(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-a")
	registerReady(t, registry, "agent-b")
	firstAgent, _ := registry.GetAgent("agent-a")
	secondAgent, _ := registry.GetAgent("agent-b")
	template := testTemplate(t)
	request := ReservationRequest{
		RunID: "run-candidate", StageExecutionID: "stage-candidate",
		Bindings: []BindingRequirement{
			testBinding(t, "first", "first", template),
			testBinding(t, "second", "second", template),
		},
	}
	edge := func(logical string, candidate AgentSnapshot) CandidateEdge {
		return CandidateEdge{
			LogicalAgentName: logical, RuntimeAgentID: candidate.Principal.RuntimeAgentID,
			RuntimeAgentInstanceID:    candidate.Registration.InstanceID,
			RuntimeAgentLabelRevision: candidate.Principal.LabelRevision,
			RequiredRuntimeAdapters:   []contracts.RuntimeAdapterRef{},
		}
	}
	reservations, err := registry.ReserveCandidateEdges(request, []CandidateEdge{
		edge("first", firstAgent), edge("first", secondAgent), edge("second", firstAgent),
	})
	if err != nil {
		t.Fatal(err)
	}
	assigned := map[string]string{}
	for _, reservation := range reservations {
		assigned[reservation.Grant.LogicalAgentName] = reservation.Grant.RuntimeInstanceID
	}
	if assigned["first"] != "agent-b" || assigned["second"] != "agent-a" {
		t.Fatalf("candidate assignment = %v", assigned)
	}
	if _, err := registry.GetStageReservations(request.StageExecutionID); !errors.Is(err, ErrReservationConflict) {
		t.Fatalf("provisional reservation visibility error = %v", err)
	}
	if err := registry.DiscardCandidateReservations(request.StageExecutionID); err != nil {
		t.Fatal(err)
	}
}

func TestReserveCandidateEdgesLeavesEverySlotFreeWithoutCompleteMatching(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-a")
	registerReady(t, registry, "agent-b")
	firstAgent, _ := registry.GetAgent("agent-a")
	template := testTemplate(t)
	request := ReservationRequest{
		RunID: "run-incomplete-candidate", StageExecutionID: "stage-incomplete-candidate",
		Bindings: []BindingRequirement{
			testBinding(t, "first", "first", template),
			testBinding(t, "second", "second", template),
		},
	}
	edge := func(logical string) CandidateEdge {
		return CandidateEdge{
			LogicalAgentName: logical, RuntimeAgentID: firstAgent.Principal.RuntimeAgentID,
			RuntimeAgentInstanceID:    firstAgent.Registration.InstanceID,
			RuntimeAgentLabelRevision: firstAgent.Principal.LabelRevision,
			RequiredRuntimeAdapters:   []contracts.RuntimeAdapterRef{},
		}
	}
	if _, err := registry.ReserveCandidateEdges(request, []CandidateEdge{edge("first"), edge("second")}); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("incomplete candidate matching error = %v", err)
	}
	for _, instanceID := range []string{"agent-a", "agent-b"} {
		agent, err := registry.GetAgent(instanceID)
		if err != nil || agent.AuthoritativeAllocationID != nil {
			t.Fatalf("failed matching mutated %s: (%+v, %v)", instanceID, agent, err)
		}
	}
}

func TestReserveAllRejectsRunReservedAgentNamespace(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	_, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-1",
		Bindings: []BindingRequirement{testBinding(t, "builder", "inputs", testTemplate(t))},
	})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("reserved namespace error = %v", err)
	}
}

func TestConcurrentReserveAllNeverAssignsSlotTwice(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	registerReady(t, registry, "agent-2")
	template := testTemplate(t)
	bindings := []BindingRequirement{
		testBinding(t, "first", "first", template),
		testBinding(t, "second", "second", template),
	}
	type result struct {
		reservations []Reservation
		err          error
	}
	results := make(chan result, 2)
	var start sync.WaitGroup
	start.Add(1)
	for index := range 2 {
		go func(index int) {
			start.Wait()
			reservations, err := registry.ReserveAll(ReservationRequest{
				RunID: fmt.Sprintf("run-%d", index), StageExecutionID: fmt.Sprintf("stage-%d", index), Bindings: bindings,
			})
			results <- result{reservations, err}
		}(index)
	}
	start.Done()
	var success result
	failures := 0
	for range 2 {
		current := <-results
		if current.err == nil {
			success = current
		} else if errors.Is(current.err, ErrInsufficientCapacity) {
			failures++
		} else {
			t.Fatalf("unexpected reservation error: %v", current.err)
		}
	}
	if len(success.reservations) != 2 || failures != 1 {
		t.Fatalf("concurrent outcomes = success %+v, failures %d", success, failures)
	}
	if success.reservations[0].Grant.RuntimeInstanceID == success.reservations[1].Grant.RuntimeInstanceID {
		t.Fatal("one Runtime Agent slot was assigned twice")
	}
}

func TestReservationRetryReturnsSameAllocationsAndGrantLifecycleIsExact(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	registerReady(t, registry, "agent-2")
	request := ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-1",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	}
	first, err := registry.ReserveAll(request)
	if err != nil {
		t.Fatal(err)
	}
	retry, err := registry.ReserveAll(request)
	if err != nil || retry[0].Grant.AllocationID != first[0].Grant.AllocationID || retry[0].Grant.RuntimeInstanceID != first[0].Grant.RuntimeInstanceID {
		t.Fatalf("reservation retry = (%+v, %v), first %+v", retry, err, first)
	}
	allocationID := first[0].Grant.AllocationID
	grant, err := registry.GetGrant(allocationID)
	if err != nil || grant.ReadPolicy != ReadCurrentRun || grant.WritePolicy != WriteInputsAndIntermediates || grant.WriteFenced {
		t.Fatalf("active grant = (%+v, %v)", grant, err)
	}
	if err := registry.SetWriteFence("wrong-allocation"); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatalf("wrong fence error = %v", err)
	}
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	grant, _ = registry.GetGrant(allocationID)
	if !grant.WriteFenced {
		t.Fatal("write fence was not visible in active grant")
	}
	if err := registry.Release("wrong-allocation"); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatalf("wrong release error = %v", err)
	}
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.GetGrant(allocationID); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatalf("released grant lookup error = %v", err)
	}
	if _, err := registry.ReserveAll(request); !errors.Is(err, ErrReservationReleased) {
		t.Fatalf("released reservation retry error = %v", err)
	}
}

func TestWriteFencedMatchingAllocationHeartbeatAlwaysDrains(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("agent-fenced-drain")
	registerReadyWith(t, registry, registration)
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-fenced-drain", StageExecutionID: "stage-fenced-drain",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	allocated := func(sequence, echoed uint64) contracts.AgentHeartbeat {
		return contracts.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: registration.InstanceID,
			HeartbeatSeq: sequence, EchoedAckSeq: echoed,
			ObservedState: contracts.AgentAllocated, AllocationID: &allocationID,
		}
	}
	if response, err := registry.Heartbeat(allocated(3, 2)); err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("active allocation heartbeat = (%+v, %v)", response, err)
	}
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	if response, err := registry.Heartbeat(allocated(4, 3)); err != nil || response.Action != contracts.ActionDrain {
		t.Fatalf("write-fenced heartbeat = (%+v, %v)", response, err)
	}

	registration.ObservedState = contracts.AgentAllocated
	registration.AllocationID = &allocationID
	if snapshot, err := registry.Register(registration); err != nil || !snapshot.ReconciliationRequired {
		t.Fatalf("write-fenced re-registration = (%+v, %v)", snapshot, err)
	}
	if response, err := registry.Heartbeat(allocated(5, 4)); err != nil || response.Action != contracts.ActionDrain {
		t.Fatalf("heartbeat after re-registration = (%+v, %v)", response, err)
	}
}

func TestReservationFingerprintPinsExactResolvedSkillsManifest(t *testing.T) {
	t.Parallel()

	registry := newTestRegistry(t, newTestClock())
	registerReady(t, registry, "agent-skill")
	template := testTemplate(t)
	template.Skills = []contracts.ArtifactRef{{Namespace: contracts.AgentSkillNamespace, Name: "review"}}
	revision := "run-review-1"
	binding := testBinding(t, "builder", "builder", template)
	binding.ResolvedSkills = []contracts.ResolvedSkill{{
		Name: "review",
		Artifact: contracts.ArtifactRef{
			Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &revision,
		},
		PackageDigest: "sha256:" + strings.Repeat("a", 64),
	}}
	request := ReservationRequest{
		RunID: "run-skill", StageExecutionID: "stage-skill",
		Bindings: []BindingRequirement{binding},
	}
	first, err := registry.ReserveAll(request)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := registry.ReserveAll(request)
	if err != nil || replayed[0].Grant.AllocationID != first[0].Grant.AllocationID {
		t.Fatalf("exact Skill manifest did not replay: (%+v, %v)", replayed, err)
	}

	changed := request
	changed.Bindings = append([]BindingRequirement(nil), request.Bindings...)
	changed.Bindings[0].ResolvedSkills = contracts.CloneResolvedSkills(binding.ResolvedSkills)
	changed.Bindings[0].ResolvedSkills[0].PackageDigest = "sha256:" + strings.Repeat("b", 64)
	if _, err := registry.ReserveAll(changed); !errors.Is(err, ErrReservationConflict) {
		t.Fatalf("substituted Skill digest replay error = %v", err)
	}
	newRevision := "run-review-2"
	changed.Bindings[0].ResolvedSkills[0].PackageDigest = binding.ResolvedSkills[0].PackageDigest
	changed.Bindings[0].ResolvedSkills[0].Artifact.Revision = &newRevision
	if _, err := registry.ReserveAll(changed); !errors.Is(err, ErrReservationConflict) {
		t.Fatalf("substituted Skill revision replay error = %v", err)
	}
	if first[0].ResolvedSkills[0].Artifact.Revision == nil ||
		*first[0].ResolvedSkills[0].Artifact.Revision != revision {
		t.Fatalf("returned reservation aliases caller manifest: %+v", first[0].ResolvedSkills)
	}
}

func TestWriteFenceWaitsForAuthorizedArtifactMutation(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registerReady(t, registry, "agent-write-gate")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-write-gate", StageExecutionID: "stage-write-gate",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	writeStarted := make(chan struct{})
	finishWrite := make(chan struct{})
	writeDone := make(chan error, 1)
	go func() {
		writeDone <- registry.WithWriteGrant(allocationID, func(grant AllocationGrant) error {
			if grant.WriteFenced {
				return errors.New("write unexpectedly started fenced")
			}
			close(writeStarted)
			<-finishWrite
			return nil
		})
	}()
	<-writeStarted
	fenceDone := make(chan error, 1)
	go func() { fenceDone <- registry.SetWriteFence(allocationID) }()
	select {
	case err := <-fenceDone:
		t.Fatalf("write fence overtook in-flight mutation: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	close(finishWrite)
	if err := <-writeDone; err != nil {
		t.Fatal(err)
	}
	if err := <-fenceDone; err != nil {
		t.Fatal(err)
	}
	if err := registry.WithWriteGrant(allocationID, func(grant AllocationGrant) error {
		if !grant.WriteFenced {
			return errors.New("post-fence mutation observed an unfenced grant")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestReleasedStageReservationHistoryIsCompactAndBounded(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registry.mu.Lock()
	for index := 0; index <= stageReservationTombstoneLimit; index++ {
		stageID := fmt.Sprintf("stage-tombstone-%d", index)
		registry.stageReservations[stageID] = stageReservation{
			fingerprint: "sha256:bounded", allocationIDs: []string{},
		}
		registry.compactStageReservationLocked(stageID)
	}
	count := len(registry.stageReservations)
	_, oldestPresent := registry.stageReservations["stage-tombstone-0"]
	newest := registry.stageReservations[fmt.Sprintf("stage-tombstone-%d", stageReservationTombstoneLimit)]
	registry.mu.Unlock()
	if count != stageReservationTombstoneLimit || oldestPresent || !newest.released || newest.allocationIDs != nil {
		t.Fatalf("bounded reservation tombstones = count:%d oldest:%v newest:%+v", count, oldestPresent, newest)
	}

	fingerprint, _, err := normalizeReservationRequest(ReservationRequest{
		RunID: "run-fingerprint", StageExecutionID: "stage-fingerprint",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil || len(fingerprint) != len("sha256:")+64 {
		t.Fatalf("compact reservation fingerprint = (%q, %v)", fingerprint, err)
	}
}

func TestCapabilityMatchingRequiresExactRefsAndSelectedTools(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*contracts.ResolvedAgentTemplate)
	}{
		{"runtime", func(template *contracts.ResolvedAgentTemplate) {
			template.Runtime.Version = "2"
		}},
		{"sandbox", func(template *contracts.ResolvedAgentTemplate) {
			template.SandboxProfile.Version = "2"
		}},
		{"toolset", func(template *contracts.ResolvedAgentTemplate) {
			template.Toolsets[0].Ref.Version = "2"
		}},
		{"selected-tool", func(template *contracts.ResolvedAgentTemplate) {
			template.Toolsets[0].Tools = append(template.Toolsets[0].Tools, "unavailable_tool")
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registry := newTestRegistry(t, newTestClock())
			registerReady(t, registry, "agent-1")
			template := testTemplate(t)
			test.mutate(&template)
			before := registry.SnapshotOperations()
			_, err := registry.ReserveAll(ReservationRequest{
				RunID: "run-1", StageExecutionID: "stage-1",
				Bindings: []BindingRequirement{testBinding(t, "builder", "builder", template)},
			})
			if !errors.Is(err, ErrInsufficientCapacity) {
				t.Fatalf("incompatible %s reservation error = %v", test.name, err)
			}
			after := registry.SnapshotOperations()
			if after.Cursor != before.Cursor || len(after.Allocations) != 0 ||
				after.RuntimeAgents[0].AuthoritativeAllocationID != nil {
				t.Fatalf("failed %s match mutated Operations: before=%+v after=%+v", test.name, before, after)
			}
		})
	}
}

func TestInjectedOrderingIsDeterministic(t *testing.T) {
	clock := newTestClock()
	var counter atomic.Uint64
	registry, err := NewRegistry(RegistryOptions{
		Now: clock.Now, MonotonicNow: clock.MonotonicNow,
		NewID: func(prefix string) (string, error) {
			return fmt.Sprintf("%s%d", prefix, counter.Add(1)), nil
		},
		AgentOrderKey: func(registration contracts.AgentRegistration) string {
			if registration.InstanceID == "agent-2" {
				return "first"
			}
			return "second"
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	registerReady(t, registry, "agent-1")
	registerReady(t, registry, "agent-2")
	reservation, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-order", StageExecutionID: "stage-order",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil || reservation[0].Grant.RuntimeInstanceID != "agent-2" {
		t.Fatalf("ordered reservation = (%+v, %v)", reservation, err)
	}
}

func TestRegistrationNeverAdoptsObservedAllocation(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("agent-orphan")
	orphan := "allocation-from-old-control-plane"
	registration.ObservedState = contracts.AgentFenced
	registration.AllocationID = &orphan
	snapshot, err := registry.Register(registration)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.AuthoritativeAllocationID != nil || !snapshot.ReconciliationRequired {
		t.Fatalf("orphan registration was adopted: %+v", snapshot)
	}
	heartbeatRequest := contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: registration.InstanceID,
		HeartbeatSeq: 1, ObservedState: contracts.AgentFenced, AllocationID: &orphan,
	}
	response, err := registry.Heartbeat(heartbeatRequest)
	if err != nil || response.Action != contracts.ActionRelease || response.AllocationID == nil || *response.AllocationID != orphan {
		t.Fatalf("orphan reconciliation response = (%+v, %v)", response, err)
	}
}

func TestRegistrationResponseLossRetryIsIdempotent(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registration := testRegistration("agent-retry")
	first, err := registry.Register(registration)
	if err != nil {
		t.Fatal(err)
	}
	clock.Advance(time.Second)
	replayed, err := registry.Register(registration)
	if err != nil || replayed.Registration.InstanceID != first.Registration.InstanceID ||
		replayed.AuthoritativeAllocationID != nil {
		t.Fatalf("registration retry = (%+v, %v), first %+v", replayed, err, first)
	}
	changed := registration
	changed.ControlURL = "https://different.example:9443"
	if _, err := registry.Register(changed); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("changed registration identity error = %v", err)
	}
}

func TestAuthenticatedPrincipalOwnsOnlyOneLiveProcessAndEveryHeartbeat(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	principal := AuthenticatedPrincipal{
		RuntimeAgentID: strings.Repeat("a", 64), Labels: []string{"debug"}, LabelRevision: 3,
	}
	first := testRegistrationV2("agent-principal-first")
	first.InitialLabels = []string{"startup-value"}
	if _, err := registry.RegisterAuthenticated(principal, first); err != nil {
		t.Fatal(err)
	}

	retry := first
	retry.InitialLabels = []string{}
	snapshot, err := registry.RegisterAuthenticated(principal, retry)
	if err != nil || snapshot.Principal.LabelRevision != 3 ||
		!equalTestStrings(snapshot.Principal.Labels, []string{"debug"}) {
		t.Fatalf("same-process retry = (%+v, %v)", snapshot, err)
	}
	second := testRegistrationV2("agent-principal-second")
	if _, err := registry.RegisterAuthenticated(principal, second); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("concurrent same-principal registration error = %v", err)
	}
	if _, err := registry.HeartbeatAuthenticated(
		strings.Repeat("b", 64), heartbeat(first.InstanceID, 1, 0),
	); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("foreign-principal heartbeat error = %v", err)
	}
	if _, err := registry.HeartbeatAuthenticated(
		principal.RuntimeAgentID, heartbeat(first.InstanceID, 1, 0),
	); err != nil {
		t.Fatal(err)
	}

	clock.Advance(61 * time.Second)
	registry.PollAllocationLosses()
	if _, err := registry.RegisterAuthenticated(principal, second); err != nil {
		t.Fatalf("post-expiry same-principal registration: %v", err)
	}
}

func TestPrincipalDeletionGuardExcludesRegistrationWithoutHoldingDurableWork(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	principal := AuthenticatedPrincipal{
		RuntimeAgentID: strings.Repeat("c", 64), Labels: []string{}, LabelRevision: 1,
	}
	registration := testRegistrationV2("agent-delete-guard")
	if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.BeginPrincipalDeletion(principal.RuntimeAgentID); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("live principal deletion guard error = %v", err)
	}
	clock.Advance(61 * time.Second)
	registry.PollAllocationLosses()
	release, err := registry.BeginPrincipalDeletion(principal.RuntimeAgentID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.RegisterAuthenticated(principal, registration); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("registration during deletion error = %v", err)
	}
	release()
	if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
		t.Fatalf("registration after deletion guard release: %v", err)
	}
}

func equalTestStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func TestLeaseExpiryAfterResponsePartitionIsIrreversible(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-lease", StageExecutionID: "stage-lease",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	// Ack 2 was received before the simulated response-only partition. Every
	// later request repeats it and therefore cannot renew the confirmed lease.
	allocated := func(sequence, echoed uint64) contracts.AgentHeartbeat {
		return contracts.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: "agent-1",
			HeartbeatSeq: sequence, EchoedAckSeq: echoed,
			ObservedState: contracts.AgentAllocated, AllocationID: &allocationID,
		}
	}
	if _, err := registry.Heartbeat(allocated(3, 2)); err != nil {
		t.Fatal(err)
	}
	for sequence := uint64(4); sequence <= 8; sequence++ {
		clock.Advance(10 * time.Second)
		if _, err := registry.Heartbeat(allocated(sequence, 2)); err != nil {
			t.Fatal(err)
		}
	}
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("lease expired early: %+v", losses)
	}
	clock.Advance(10 * time.Second)
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].AllocationID != allocationID ||
		losses[0].Reason != LossControlLeaseExpired {
		t.Fatalf("lease losses = %+v", losses)
	}
	grant, err := registry.GetGrant(allocationID)
	if err != nil || !grant.Lost || !grant.WriteFenced {
		t.Fatalf("lost grant = (%+v, %v)", grant, err)
	}
	// Ack 7 was issued while responses were supposedly lost. Even a delayed
	// echo cannot revive the already-lost allocation.
	if response, err := registry.Heartbeat(allocated(9, 7)); err != nil || response.Action != contracts.ActionDrain {
		t.Fatalf("late ack response = (%+v, %v)", response, err)
	}
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("loss edge was emitted more than once: %+v", losses)
	}
	snapshot, _ := registry.GetAgent("agent-1")
	if !snapshot.LeaseExpired || !snapshot.ReconciliationRequired {
		t.Fatalf("expired agent snapshot = %+v", snapshot)
	}
}

func TestReservationAllowsIdleOnlyUntilAllocationIsObservedActive(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-prepare", StageExecutionID: "stage-prepare",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID

	response, err := registry.Heartbeat(heartbeat("agent-1", 3, 2))
	if err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("idle preparation transition = (%+v, %v)", response, err)
	}
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("reservation-to-prepare transition was lost: %+v", losses)
	}
	active := contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: "agent-1",
		HeartbeatSeq: 4, EchoedAckSeq: 3,
		ObservedState: contracts.AgentAllocated, AllocationID: &allocationID,
	}
	if response, err = registry.Heartbeat(active); err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("allocation activation = (%+v, %v)", response, err)
	}
	response, err = registry.Heartbeat(heartbeat("agent-1", 5, 4))
	if err != nil || response.Action != contracts.ActionDrain {
		t.Fatalf("post-activation idle mismatch = (%+v, %v)", response, err)
	}
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].Reason != LossRuntimeMismatch {
		t.Fatalf("post-activation losses = %+v", losses)
	}
}

func TestLeaseUsesMonotonicClockNotWallClock(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")

	clock.JumpWall(24 * time.Hour)
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("wall-clock jump expired lease: %+v", losses)
	}
	clock.JumpWall(-48 * time.Hour)
	clock.Advance(time.Minute)
	registry.PollAllocationLosses()
	if _, err := registry.GetAgent("agent-1"); !errors.Is(err, ErrAgentNotFound) {
		t.Fatalf("allocation-free expired process was not retired: %v", err)
	}
}

func TestRuntimeRegistrationCapacityIgnoresRetiredProcessHistory(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	firstInstanceID := "retired-agent-00000"
	for index := 0; index <= maximumOperationsItems; index++ {
		instanceID := fmt.Sprintf("retired-agent-%05d", index)
		if _, err := registry.Register(testRegistration(instanceID)); err != nil {
			t.Fatalf("register process %d: %v", index, err)
		}
		clock.Advance(time.Minute)
	}
	if count := len(registry.agents); count != 1 {
		t.Fatalf("retained process entries = %d, want 1", count)
	}
	operations := registry.SnapshotOperations()
	if len(operations.RuntimeAgents) != 1 || operations.RuntimeAgents[0].InstanceID != "retired-agent-10000" {
		t.Fatalf("current Runtime Agents = %+v", operations.RuntimeAgents)
	}
	response, err := registry.Heartbeat(heartbeat(firstInstanceID, 1, 0))
	if err != nil || response.Action != contracts.ActionReregister {
		t.Fatalf("retired process heartbeat = (%+v, %v)", response, err)
	}
}

func TestExpiredAllocationOwnerIsRetainedUntilRelease(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "expired-owner")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-expired-owner", StageExecutionID: "stage-expired-owner",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	clock.Advance(time.Minute)
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].AllocationID != allocationID {
		t.Fatalf("allocation losses = %+v", losses)
	}
	if _, present := registry.agents["expired-owner"]; !present {
		t.Fatal("expired process disappeared while it still owned an allocation")
	}
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	if _, present := registry.agents["expired-owner"]; present {
		t.Fatal("expired process remained after authoritative release")
	}
}

func TestAllocationFreeSupersededProcessIsRetiredImmediately(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	oldRegistration := testRegistration("superseded-idle")
	registerReadyWith(t, registry, oldRegistration)
	replacement := testRegistration("replacement-idle")
	replacement.ControlURL = oldRegistration.ControlURL
	replacement.A2AURL = oldRegistration.A2AURL
	if _, err := registry.Register(replacement); err != nil {
		t.Fatal(err)
	}
	if _, present := registry.agents[oldRegistration.InstanceID]; present {
		t.Fatal("allocation-free superseded process remained in the live Registry")
	}
	if _, present := registry.agents[replacement.InstanceID]; !present {
		t.Fatal("replacement process is absent")
	}
	response, err := registry.Heartbeat(heartbeat(oldRegistration.InstanceID, 3, 2))
	if err != nil || response.Action != contracts.ActionReregister {
		t.Fatalf("superseded process heartbeat = (%+v, %v)", response, err)
	}
}

func TestReconcileRuntimeRestartWithholdsNewInstanceUntilOldAllocationReleased(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	oldRegistration := testRegistration("agent-old")
	registerReadyWith(t, registry, oldRegistration)
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-old", StageExecutionID: "stage-old",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}

	restarted := testRegistration("agent-new")
	restarted.ControlURL = oldRegistration.ControlURL
	restarted.A2AURL = oldRegistration.A2AURL
	registerReadyWith(t, registry, restarted)
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].Reason != LossRuntimeRestarted {
		t.Fatalf("restart losses = %+v", losses)
	}
	_, err = registry.ReserveAll(ReservationRequest{
		RunID: "run-new", StageExecutionID: "stage-new",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("new instance was offered before reconciliation: %v", err)
	}
	if err := registry.Release(reservations[0].Grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	available, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-new", StageExecutionID: "stage-new-after-release",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil || available[0].Grant.RuntimeInstanceID != "agent-new" {
		t.Fatalf("new instance after reconciliation = (%+v, %v)", available, err)
	}
}

func TestReconcileLostReleaseResponseRepeatsReleaseWithoutSlotReuse(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-release", StageExecutionID: "stage-release",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	fenced := func(sequence, echoed uint64) contracts.AgentHeartbeat {
		return contracts.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: "agent-1",
			HeartbeatSeq: sequence, EchoedAckSeq: echoed,
			ObservedState: contracts.AgentFenced, AllocationID: &allocationID,
		}
	}
	for sequence := uint64(3); sequence <= 4; sequence++ {
		response, err := registry.Heartbeat(fenced(sequence, sequence-1))
		if err != nil || response.Action != contracts.ActionRelease ||
			response.AllocationID == nil || *response.AllocationID != allocationID {
			t.Fatalf("release reconciliation %d = (%+v, %v)", sequence, response, err)
		}
	}
	if _, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-too-early", StageExecutionID: "stage-too-early",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	}); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("fenced slot was reused: %v", err)
	}
	response, err := registry.Heartbeat(heartbeat("agent-1", 5, 4))
	if err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("confirmed idle reconciliation = (%+v, %v)", response, err)
	}
}

func TestReleaseDoesNotReuseSlotFromStaleIdleHeartbeat(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-short", StageExecutionID: "stage-short",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	// No allocated heartbeat arrived: the registry's latest observation is the
	// idle heartbeat from before the private prepare call.
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	snapshot, err := registry.GetAgent("agent-1")
	if err != nil || snapshot.AuthoritativeAllocationID != nil || !snapshot.ReconciliationRequired {
		t.Fatalf("released stale-idle slot = (%+v, %v)", snapshot, err)
	}
	if _, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-too-early", StageExecutionID: "stage-too-early-stale-idle",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	}); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("stale idle observation permitted immediate slot reuse: %v", err)
	}

	response, err := registry.Heartbeat(heartbeat("agent-1", 3, 2))
	if err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("post-release idle confirmation = (%+v, %v)", response, err)
	}
	if _, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-after-confirm", StageExecutionID: "stage-after-idle-confirm",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	}); err != nil {
		t.Fatalf("confirmed idle slot was not reusable: %v", err)
	}
}

func testTemplate(t *testing.T) contracts.ResolvedAgentTemplate {
	t.Helper()
	snapshot, err := config.Load("../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	return template
}

func likeC4ValidationTemplate(t *testing.T) contracts.ResolvedAgentTemplate {
	t.Helper()
	template := testTemplate(t)
	template.Toolsets = append(template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "likec4", Version: "1"},
		Tools: []string{"validate_likec4"},
	})
	return template
}

func codeAnalysisTemplate(t *testing.T, tool string) contracts.ResolvedAgentTemplate {
	t.Helper()
	template := testTemplate(t)
	template.Toolsets = append(template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "code-analysis", Version: "1"},
		Tools: []string{tool},
	})
	return template
}

func testBinding(
	t *testing.T,
	logicalAgentName string,
	namespace string,
	template contracts.ResolvedAgentTemplate,
) BindingRequirement {
	t.Helper()
	snapshot, err := config.Load("../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	return BindingRequirement{
		LogicalAgentName: logicalAgentName,
		Namespace:        namespace,
		AgentTemplate:    template,
		ExecutionConfig: AllocationExecutionConfig{
			ModelPolicy: template.ModelPolicy.Ref,
			LLMGateway:  gateway.Ref,
		},
	}
}

func testRegistration(instanceID string) contracts.AgentRegistration {
	return contracts.AgentRegistration{
		APIVersion: contracts.APIVersion, InstanceID: instanceID, SoftwareVersion: "0.1.0",
		StartedAt:         time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC),
		ControlURL:        "https://" + instanceID + ".example:9443",
		A2AURL:            "https://" + instanceID + ".example:9444",
		SupportedRuntimes: []string{"adk@1"},
		SupportedToolsets: []contracts.ToolsetCapability{{
			Ref: "run-artifacts@1", Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
		}},
		SupportedSandboxProfiles: []string{"local-workdir@1"}, ObservedState: contracts.AgentIdle,
	}
}

func heartbeat(instanceID string, sequence, echoed uint64) contracts.AgentHeartbeat {
	return contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: instanceID,
		HeartbeatSeq: sequence, EchoedAckSeq: echoed, ObservedState: contracts.AgentIdle,
	}
}

func registerReady(t *testing.T, registry *InMemoryRegistry, instanceID string) {
	t.Helper()
	registerReadyWith(t, registry, testRegistration(instanceID))
}

func registerReadyWith(t *testing.T, registry *InMemoryRegistry, registration contracts.AgentRegistration) {
	t.Helper()
	if _, err := registry.Register(registration); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(heartbeat(registration.InstanceID, 1, 0)); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(heartbeat(registration.InstanceID, 2, 1)); err != nil {
		t.Fatal(err)
	}
}

func newTestRegistry(t *testing.T, clock *testClock) *InMemoryRegistry {
	t.Helper()
	var counter atomic.Uint64
	registry, err := NewRegistry(RegistryOptions{
		Now: clock.Now, MonotonicNow: clock.MonotonicNow,
		NewID: func(prefix string) (string, error) {
			return fmt.Sprintf("%s%d", prefix, counter.Add(1)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

type testClock struct {
	mu        sync.Mutex
	now       time.Time
	monotonic time.Duration
}

func newTestClock() *testClock {
	return &testClock{now: time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)}
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *testClock) Advance(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(duration)
	c.monotonic += duration
}

func (c *testClock) JumpWall(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(duration)
}

func (c *testClock) MonotonicNow() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.monotonic
}
