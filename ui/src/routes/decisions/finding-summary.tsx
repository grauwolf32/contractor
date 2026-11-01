import type {ProjectFinding} from "../../api/audits";
import { RecordedTime } from "../../app/recorded-time";
import { findingStateLabel,severityLabel,verdictLabel } from "../../app/vocabulary";
import { Facts,StatusChip } from "../../ui";
import { FindingLocations } from "../projects/audits/finding-locations";
import { DecisionMarkdown } from "./markdown";
import "./decisions.css";

export function FindingSummary({finding,titleAs:Title="h2",variant="full"}:{finding:ProjectFinding;auditId?:string;variant?:"full"|"compact";titleAs?:"h2"|"h3"}) {
 const claim=finding.claim;const state=findingStateLabel(finding.state);
 return <section className="decisions-finding-summary">
 <Title>{finding.title}</Title>
 <StatusChip tone={state.tone}>{state.label}</StatusChip>
 <Facts items={[
 {term:"Analyst verdict",value:verdictLabel(finding.analystVerdict).label},
 {term:"Severity",value:severityLabel(finding.analystSeverity)},
 {term:"Found",value:<RecordedTime value={finding.createdAt}/>}
 ]}/>
 {claim===undefined?null:<>
 <DecisionMarkdown source={claim.description}/>
 {variant==="compact"?null:<>
 {claim.hypothesis===undefined || claim.hypothesis===""?null:<section><h3>Hypothesis</h3><DecisionMarkdown source={claim.hypothesis}/></section>}
 {claim.preconditions.length===0?null:<section><h3>Conditions</h3><ul>{claim.preconditions.map((text,index)=><li key={index}>{text}</li>)}</ul></section>}
 {claim.locations===undefined?null:<FindingLocations document={claim}/>}
 {claim.severity_suggestion===""?null:<p className="ui-caption">Source severity suggestion: {severityLabel(claim.severity_suggestion)}</p>}
 {claim.limitations.length===0?null:<section><h3>Limitations</h3><ul>{claim.limitations.map((text,index)=><li key={index}>{text}</li>)}</ul></section>}
 </>}
 </>}
 {finding.rationale===null?null:<section><h3>Current assessment</h3><DecisionMarkdown source={finding.rationale}/><p className="ui-caption">{finding.actorId}{finding.assessedAt===null?null:<> · <RecordedTime value={finding.assessedAt}/></>}</p></section>}
 </section>;
}
