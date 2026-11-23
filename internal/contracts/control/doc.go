// Package control holds the private Control Plane/Runtime Agent wire
// contracts: Runtime Agent registration and heartbeats, the AllocationSpec
// and the requests of the allocation lifecycle.
//
// An AllocationSpec composes the other wire contracts (the resolved
// AgentTemplate, Skills, workspace, Runtime settings and their provenance,
// Run metadata labels and the performance request), so control sits above
// those packages and none of them imports it.
package control
