# LikeC4 modeling for threat analysis

Load this reference only when its topic becomes relevant, with
`load_skill_resource(skill_name="stride", file_path="references/likec4-modeling.md")`.

LikeC4 helps align architecture, flows and trust boundaries, which are then
used to work through threats. A useful model shows who controls each part,
what they are entitled to do, and which rule an attacker could break. Diagrams
alone are not enough: invariants, threat scenarios, assumptions and residual
risk decisions belong alongside them.

The starting set is the system context, the main flows, authorization and one
critical scenario. Add the other views for a specific question. These
recommendations describe a working approach, not a mandatory LikeC4 notation.

Writing `.c4` sources and running `likec4 validate` are authoring steps.
Perform them only when the current Worker invocation actually exposes
workspace writing and the relevant execution operations. When the invocation
only reads or discusses models, use the interpretation guidance below,
propose the missing model as follow-up work, and record authoring as an
unavailable capability. This reference grants no file, network or execution
authority (see the LikeC4 language reference and CLI docs at
https://likec4.dev/dsl/intro/ and https://likec4.dev/tooling/cli/).

## Where to start

Fix the subject of analysis first: an existing implementation or a designed
architecture, the functions in scope, external dependencies and assumptions.
For an existing system, state what confirms the behavior of the essential
parts. For a design, separate required behavior from mechanisms that already
exist.

Before drawing anything, write down:

- **Assets**: confidential data, money, the right to perform operations,
  production availability, keys and proof of execution.
- **Invariants**: who may read an object; which version may be executed;
  which event grants a payout right; which obligations the guarantee covers.
- **Participants**: users and their roles, organizations, services and
  devices. Consider abuse of legitimate authority, compromise and collusion,
  not only stolen credentials.
- **Open questions**: unknown checks, how keys are issued, rules for
  cancellation and change of assignments, handling of late-arriving events.

An example invariant: "a subcontractor may complete only the operation
assigned to them". Such an invariant suggests the required elements and
checks more precisely than the original requirement "add API protection".
Decomposing the system, analyzing threats and choosing responses are
sequential parts of threat modeling (see OWASP Threat Modeling and the
Microsoft SDL threat-modeling guidance).

## Which components to include

Add an element when control, authority, state, untrusted-data handling or the
decision taken differ materially for it. If the difference influences a
threat scenario, it deserves a place in the model.

| Category | What to include | Why |
| --- | --- | --- |
| Users and organizations | Roles, user browsers, customers, contractors | Show original rights and independent parties |
| Identification and authorization | IdP, server-side session verification, service identity, device registry | Understand who establishes identity and who permits the action |
| Request intake | API, webhooks, imports, admin interfaces | Record where commands and data arrive |
| Critical decisions | Version approval, delegation, payout rights | Show where a business rule is applied |
| Data processing | Parsers, workers, converters, job executors | Investigate untrusted input and the authority available to the process |
| State and asynchrony | Databases, object stores, caches, queues, retries | Track saved context and later execution |
| Secrets and proof | Key stores, audit logs, measurement reports | Define who can use, change or substitute them |
| Periphery and external systems | Gateway, machine tool, offline cache, bank, external organization | Show the parts outside the platform's control |

Distinguish placement from responsibility. A "portal" may mean browser code
and a server part; separate them, or user input will look like trusted
server state. A functional module can be extracted for decision analysis even
when it runs in the same process as the rest of the API; extracting it does
not declare a separate microservice or isolation.

Not every library, table, function or replica needs listing. Add detail when
a specific threat depends on it: shared worker credentials, a dangerous
parser, or a privileged recovery procedure.

For an essential element, describe the owner, runtime identity, permission
scope, accessible data, inputs and the effects of its actions. A key is an
asset; a key store is a storage component; key usage is an interaction. Do
not mix these entities.

## How to draw trust boundaries

A boundary separates areas with different control, authority or grounds for
trusting specific data. A flow shows the boundary crossing; identity helps
determine the participant and the limits of their rights. Assess trust per
action: a gateway may be authorized to receive CAD, but its message about a
part's quality needs a separate basis for trust.

Useful distinctions:

- **Administrative domains**: the platform, the customer, the factory, an
  independent IdP.
- **Execution constraints**: a job worker, a sandbox, a privileged process,
  host isolation.
- **Logical access scopes**: different tenants, roles, projects, orders and
  delegated operations.

For each boundary record: what is separated; which flows cross it; which
check is needed; what enforces it; and the status of that claim. For example:
"Requirement — the Worker reads only its own assignment's objects. Check —
its credentials, access policy and available volumes."

A shared VPC or a shared owner does not mean shared authority. Different
identities do not automatically require separate boundaries. Participants
with the same role may have different object scopes. If root controls the
application, restricting that root further requires a separate mechanism and
justification.

Keep fact, requirement and assumption distinguishable in descriptions or
tags. An example local legend: red dashed — separate control; amber —
required execution constraint; indigo — authorization scope; solid frame —
structural grouping. Colors are the model author's choice and are not a
LikeC4 standard.

## Which views to make

Every view must answer one stated question and keep a readable level of
detail. All views use one model: elements and relationships are re-selected
through predicates, not recreated per picture (see the LikeC4 views
documentation).

| View | Main question | What to show |
| --- | --- | --- |
| Context | Who interacts with whom, and who controls what | The system, organizations, external services, main flows and administrative boundaries |
| Data and processing flows | Where untrusted data appears, is transformed and is stored | APIs, workers, queues, stores, results, constrained execution environments |
| Identity and authorization | Who may act on which object | Identity issuance and verification, tenant context, roles, service rights, delegation |
| Critical scenario | In which order decisions are taken and effects occur | Approval, execution, event, payout; separately retries, competition and failures |
| Data lifecycle and offline | What remains after an operation ends or access is revoked | Copies, caches, local queues, validity periods, synchronization, old versions |
| System management | Who can change the system itself and its identities | Device registration, role management, keys, updates and recovery — when they are in scope |
| Deployment | Where network and process constraints are actually enforced | Hosts, networks, containers, instances and the real administrative zones |

Start with the first four questions. The lifecycle view is needed when there
is significant local state, copying or deferred processing. The management
view matters most when device registration or a role change enables
persistence. Add deployment when the infrastructure is known or the analysis
depends on it; LikeC4 supports a separate deployment model and deployment
views.

Use a dynamic view for ordered actions. When exploring abuse, label it as an
attack hypothesis. Allowed interactions and presumed attacker actions must
stay distinguishable in meaning and status. The order of arrows on a static
view does not define execution order.

Split an overloaded view by analysis question: "CAD processing" and "payout"
need different details. Component hierarchy, scenario sequence and network
placement usually read better as separate views. A reader must understand a
view from its title, description and legend without verbal explanation.

## How to define relations

A relation must explain what is transferred, on whose behalf, and what effect
it can cause. Formulations like "interacts" or "sends data" usually hide the
information the analysis needs.

| Relation kind | Example label | What to verify |
| --- | --- | --- |
| Command or query | Approve order version | Identity, role, object, transition validity |
| Data | CAD bytes of the assigned job | Confidentiality, integrity, access scope |
| Event | Job-17 completed | Source authority, relation to the job, replay, state |
| Queued job | Process tenant / job / version | Context origin, consumer rights, delay and replay |
| Identification | SSO assertion for the platform | Trusted issuer, audience, lifetime and link to local rights |
| Delegation | Permit an operation and an object set | Who can grant the right, its limits and revocation |
| Administration | Register the factory gateway | Registration right, device-organization link, later revocation |

Choose the meaning of the arrow direction. In a data-flow diagram, CAD bytes
flow from storage to the Worker; a read request goes from the Worker to the
storage. When both matter, show two flows. A job label or mTLS does not
define who initiated the network connection; polling and delivery can also
have different directions.

In the description of an essential relation, state the resource, principal,
permission scope, channel and check point. For asynchronous interaction, add
the saved context, redelivery and the conditions under which the decision
remains valid. When necessary, separate sender authentication, operation
authorization, content validation and proof of the actual result.

On a context view several interactions can merge into one meaningful label.
On a detailed view, separate them when rights, data or consequences differ.
LikeC4 may merge relations when elements are collapsed; check the resulting
labels and refine them through view predicates and overrides (see the
LikeC4 view predicates documentation).

## How to organize LikeC4 sources

Store entities and relationships once, with stable identifiers. Use the
domain's language for names on diagrams. Element kinds should help read the
model, for example `actor`, `service`, `storage`, `queue`,
`executionBoundary` and `accessScope`.

A practical layout: `specification.c4` for kinds and the legend; `model.c4`
for elements and relations; `views.c4` for views. The deployment model and
individual scenarios can move to additional `.c4` files when that improves
navigation.

The model hierarchy reflects the chosen architecture structure. A group in a
view serves the view; it cannot automatically imply IAM, a sandbox or
placement. Consider the nesting constraints of grouped elements. When one
hierarchy shows the axes poorly — ownership, placement and data access — use
separate views with explicit descriptions of those axes (see the LikeC4
groups documentation).

Keep invariants and threat records next to the model. Link a significant
threat to concrete element, relation or boundary identifiers. A threat record
needs: the attacker and their initial rights; the action; the boundary
crossing; the rule broken; the damage; the assumptions; the protection; the
residual risk and how to verify it.

## A small DSL example

The example has two administrative owners and a separately marked requirement
to constrain the Worker. The API and its state are one module because the
internal split does not affect the demonstrated question. The context view
hides processing details; the second view exposes them. A separate dynamic
view defines the sequence for handling the completion event.

```text
// Teaching example. Boundaries and checks denote model requirements.
// likec4 validate checks the DSL, not the implementation of protections.
specification {
  element domain {
    notation 'Separate administrative domain'
    style { color red; border dashed; opacity 5% }
  }
  element jobBoundary {
    notation 'Required job execution constraint'
    style { color amber; border dashed; opacity 8% }
  }
  element service { style { color blue; size small } }
  element actor { style { shape person; color slate; size small } }
  element storage { style { shape cylinder; color green; size small } }
  element queue { style { shape queue; color green; size small } }
  relationship command { color blue; line solid }
  relationship data { color green; line solid }
  relationship event { color amber; line dashed }
}

model {
  customer = actor 'Customer'
  factory = domain 'TB01 · Factory' {
    gateway = service 'Gateway' {
      description 'Factory-controlled; separate device identity.'
    }
  }
  cloud = domain 'Platform control' {
    api = service 'API and order state' {
      description 'Verifies identity, tenant, action and the assigned job.'
    }
    files = storage 'CAD files' {
      description 'Access is limited to the objects of the assigned job.'
    }
    jobs = queue 'Processing jobs' {
      description 'Server-side tenant / job / version context.'
    }
    processing = jobBoundary 'TB02 · CAD job execution' {
      description 'Requirement: separate execution and limited rights.'
      worker = service 'Worker' {
        description 'Parses untrusted CAD; receives no whole-API authority.'
      }
    }
  }

  customer .command cloud.api 'Approve a specific version'
  cloud.api .command factory.gateway 'Deliver the accepted job'
  factory.gateway .event cloud.api 'Signed completion message'
  cloud.api .data cloud.jobs 'Create job with server-side context'
  cloud.jobs .data cloud.processing.worker 'Assigned job'
  cloud.files .data cloud.processing.worker 'Permitted CAD bytes'
  cloud.processing.worker .event cloud.api 'Processing result'
}

views {
  view index {
    title 'Context and participants'
    description 'A label identifies the message source but does not prove physical execution.'
    include customer, factory, factory.gateway, cloud, cloud.api
    autoLayout LeftRight
  }
  view cad_processing {
    title 'Job data and execution'
    description 'The amber frame is an isolation requirement; its enforcement needs separate verification.'
    include cloud, cloud.api, cloud.files, cloud.jobs,
      cloud.processing, cloud.processing.worker
    autoLayout LeftRight
  }
  dynamic view completion {
    title 'Completion event review'
    description 'Ordered steps for handling the message. Physical execution needs a separate basis of trust.'
    factory.gateway -> cloud.api 'Report completion'
    cloud.api -> cloud.api 'Verify source, scope and job state'
    cloud.api -> cloud.api 'Accept or reject the event by rule'
    include factory, cloud
  }
}
```

This example illustrates the notation; it is not a security architecture.
Validating it with `likec4 validate` confirms the DSL is well formed.
The effectiveness of isolation and checks is confirmed by implementation
details and corresponding tests. Check capabilities and CLI flags against the
installed LikeC4 version.

## Model readiness checklist

- The subject of analysis, the assets and the essential invariants are clear.
- The user's browser is separated from trusted server-side decisions.
- For every essential component, the owner, identity and permission scope are
  known.
- External participants, local state and deferred processing are shown when
  they influence threats.
- Boundaries are labeled; structural frames differ from security constraints.
- Fact, requirement and assumption are distinguishable.
- Every essential arrow has a comprehensible meaning and direction.
- For the critical scenario, retries, concurrency, version change, right
  revocation and partial failure are considered.
- The model accounts for abuse of legitimate rights and collusion, not only
  credential theft.
- Every view has its own analysis question; hidden detail did not disappear
  from the shared model.
- Legend, labels and layout survive export; no accidentally merged relations
  remain inside views.
- The threats raised are linked to rules, measures and residual risk.

Working order: assets and invariants → participants and initial rights →
architecture and state → boundaries and flows → views → abuse scenarios →
measures and residual-risk verification.
