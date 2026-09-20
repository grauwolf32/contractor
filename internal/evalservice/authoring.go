package evalservice

import (
	"context"
	"crypto/rand"
	"encoding/hex"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func newID(prefix string) (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(b[:]), nil
}

func (s *Service) Create(ctx context.Context, scope evalstore.Scope, document evaldomain.Frozen, mutation evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	var input evaldomain.CreateExperiment
	if err := evaldomain.DecodeInto("CreateExperiment", document.Bytes(), &input); err != nil {
		return evalstore.Receipt{}, err
	}
	replay, err := evalstore.NewPostgresStore(s.pool).Replay(ctx, scope, "experiments", "experiment-create", mutation)
	if err != nil {
		return evalstore.Receipt{}, err
	}
	if replay != nil {
		return *replay, nil
	}
	id, err := newID("experiment-")
	if err != nil {
		return evalstore.Receipt{}, err
	}
	portable, err := newID("eval-")
	if err != nil {
		return evalstore.Receipt{}, err
	}
	var resources []evalstore.PlanResource
	if input.Registration != nil {
		portable = input.Registration.Manifest.ExperimentID
		resources, err = s.resolveExternal(ctx, scope.OwnerID, *input.Registration)
		if err != nil {
			return evalstore.Receipt{}, err
		}
	}
	var result evalstore.Receipt
	err = s.tx(ctx, func(st *evalstore.Store) error {
		var err error
		result, err = st.Create(ctx, evalstore.CreateParams{
			Scope:      scope,
			ID:         id,
			PortableID: portable,
			Document:   document,
			Mutation:   mutation,
			Resources:  resources,
		})
		return err
	})
	return result, err
}

// Registration retains the producer's attributed manifest unchanged. The
// service additionally freezes its own exact local execution snapshots; it
// never pretends these bytes were the producer's original binding document.
func (s *Service) resolveExternal(ctx context.Context, owner string, reg evaldomain.ExternalRegistration) ([]evalstore.PlanResource, error) {
	casesByMember := map[string]evaldomain.Case{}
	for _, r := range reg.Recipes {
		casesByMember[r.MemberID] = r.Case
	}
	resources := []evalstore.PlanResource{}
	for _, v := range reg.Variants {
		cases := []evaldomain.Case{}
		seen := map[string]string{}
		for _, m := range reg.Manifest.Members {
			if m.VariantID != v.ID {
				continue
			}
			c := casesByMember[m.MemberID]
			digest, err := hashJSON(c)
			if err != nil {
				return nil, err
			}
			if prior, ok := seen[c.ID]; ok {
				if prior != digest {
					return nil, evaldomain.Failure("eval_member_conflict")
				}
				continue
			}
			seen[c.ID] = digest
			cases = append(cases, c)
		}
		preflight, err := s.resolver.Resolve(ctx, owner, v, cases)
		if err != nil {
			return nil, err
		}
		for _, m := range reg.Manifest.Members {
			if m.VariantID == v.ID && m.Eligibility == "eligible" && preflight.Cases[m.CaseID].State != "eligible" {
				return nil, evaldomain.Failure("eval_not_ready")
			}
		}
		doc, err := bindingDocument(v, preflight)
		if err != nil {
			return nil, err
		}
		resources = append(resources, evalstore.PlanResource{Path: "bindings/" + v.ID + ".json", Document: doc})
	}
	return resources, nil
}

func bindingDocument(v evaldomain.Variant, p Preflight) (evaldomain.Frozen, error) {
	inputs, outputs := v.InputMapping, v.OutputMapping
	if inputs == nil {
		inputs = map[string]string{}
	}
	if outputs == nil {
		outputs = map[string]string{}
	}
	capabilities := p.Capabilities
	if capabilities == nil {
		capabilities = []string{}
	}
	raw, err := jsonBytes(portableBinding{
		SchemaVersion: portableBindingSchema,
		ID:            v.ID,
		Provider:      managedProvider,
		Capabilities:  capabilities,
		Connection:    managedConnection,
		Settings:      bindingSettings{Variant: v, Snapshot: p.Snapshot, ObservedPins: p.Pins},
		InputMapping:  inputs,
		OutputMapping: outputs,
		Normalizers:   []normalizerReference{{ID: "contractor-output@1", ImplementationSHA256: NormalizerSHA256()}},
	})
	if err != nil {
		return evaldomain.Frozen{}, err
	}
	return evaldomain.Freeze("playground.binding/v1", raw)
}

func (s *Service) Command(ctx context.Context, scope evalstore.Scope, id string, command evaldomain.Command, mutation evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	commandID, err := newID("command-")
	if err != nil {
		return evalstore.Receipt{}, err
	}
	p := evalstore.CommandParams{Scope: scope, ExperimentID: id, CommandID: commandID, Command: command, Mutation: mutation}
	if command.Kind == "duplicate" {
		p.DuplicateID, err = newID("experiment-")
		if err != nil {
			return evalstore.Receipt{}, err
		}
		p.DuplicatePortableID, err = newID("eval-")
		if err != nil {
			return evalstore.Receipt{}, err
		}
	}
	var result evalstore.Receipt
	err = s.tx(ctx, func(st *evalstore.Store) error { var err error; result, err = st.Command(ctx, p); return err })
	return result, err
}

func (s *Service) Submit(ctx context.Context, scope evalstore.Scope, id, member string, input evaldomain.Submission, mutation evaldomain.MutationIdentity) (evalstore.Receipt, error) {
	var result evalstore.Receipt
	err := s.tx(ctx, func(st *evalstore.Store) error {
		var err error
		result, err = st.Admit(ctx, evalstore.Admission{Scope: scope, ExperimentID: id, MemberID: member, PlanSHA256: input.PlanSHA256, Mutation: mutation})
		return err
	})
	return result, err
}
