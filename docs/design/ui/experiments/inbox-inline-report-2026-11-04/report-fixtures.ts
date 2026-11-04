import * as build from '/src/test/builders';
import {json,failure} from '/src/test/responses';
const sha=async text=>'sha256:'+Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),n=>n.toString(16).padStart(2,'0')).join('');
export async function reportFixtures(server,scenario){
 const documents={};
 for(const [auditId,previous] of Object.entries(server.reports)){
  const proposed=previous.status==='proposed';
  const summary=`# API review report\n\n${proposed?'Proposed report for owner review.':'Ready report of the completed check.'} This is a synthetic, bounded assessment.\n\n## Conclusion\n\nFive operations were evaluated in this check. The recorded execution outcome and the quality of the results must be reviewed separately. ${proposed?'Two possible issues are listed in this Project and remain unassessed.':'No human security or compliance certification is implied.'}\n\n## Scope and method\n\n- Input: a fixed synthetic API fixture.\n- Work: five operation-level checks.\n- Access: no live target or production service was contacted.\n\n## Recorded work\n\n| Work | Count |\n| --- | ---: |\n| Evaluated operations | 5 |\n| Unevaluated operations | 0 |\n| Confirmed issues | 0 |\n\n## Limitations\n\nThe fixture does not establish the behavior of a deployed application. Authentication middleware, database policies and environment-specific controls require their own evidence. A completed check is not evidence that a system is secure.\n\n## Follow-up\n\nRead the retained evidence and independently assess any possible issues. ${proposed?'Review the proposed conclusions before accepting or rejecting this report.':'Use the check page to inspect the recorded results and scope.'}\n\n**End of report — the complete follow-up is available here in the Inbox.**\n\nThis report is not a security or compliance certification.`;
  const machine={outcome:'completed',evaluatedOperations:5,unevaluatedOperations:0,confirmedIssues:0,unassessedProjectProposals:proposed?2:0,certification:false};
  const markdown=await sha(summary),encoded=JSON.stringify(machine,null,2);
  const value=build.conforming('AuditReport',{...previous,summary,machine,summaryArtifact:{...previous.summaryArtifact,digest:markdown,sizeBytes:new TextEncoder().encode(summary).length},machineArtifact:{...previous.machineArtifact,digest:await sha(encoded),sizeBytes:new TextEncoder().encode(encoded).length}});
  server.reports[auditId]=value;documents[auditId]=value;
 }
 if(scenario==='report-unavailable')for(const id of Object.keys(server.reports))server.reports[id]=build.report(id,'unavailable');
 if(scenario==='report-pending')for(const id of Object.keys(server.reports))server.reports[id]=build.report(id,'pending');
 if(scenario==='report-mismatch'&&server.reports.audit_wait)server.reports.audit_wait=build.conforming('AuditReport',{...server.reports.audit_wait,review:build.review('audit_wait','other_report_review','report-acceptance')});
 return {documents,response(request,url){
  if(scenario==='report-error'&&/^\/v1\/audits\/[^/]+\/report$/.test(url.pathname))return failure(503,'storage_transaction_conflict','Synthetic report read failure');
  const match=url.pathname.match(/^\/v1\/audits\/([^/]+)\/reviews\/([^/]+)$/);if(match){const review=(server.reviews[match[1]]??[]).find(r=>r.requestId===match[2]);return review?json(review):failure(404);}
  return undefined;
 }};
}
