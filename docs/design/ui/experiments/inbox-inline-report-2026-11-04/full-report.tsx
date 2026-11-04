import {Link} from 'react-router';
import type {Audit} from '/src/api/audits';
import type {Project} from '/src/api/projects';
import {usePublicAPI} from '/src/api/context';
import {DetailHeader,DetailPane} from '/src/ui';
import {RecordedTime} from '/src/app/recorded-time';
import {DetailBar,OpenLink,Recorded,type DetailContext} from '/src/routes/inbox/detail-parts';
import type {InboxRowOf} from '/src/routes/inbox/model';
import {reportPath,checkTypeLabel} from '/src/routes/inbox/present';
import {ReviewDetail as OtherReview} from '/src/routes/inbox/detail-decisions';
import {checkPath} from '/src/routes/projects/audits/check-links';
import {decisionOutcome} from '/src/routes/decisions/model';
import {showsAcceptance,useAuditReport,useReportAcceptance} from '/src/routes/reports/report-data';
import {ReportContent,ReportAcceptanceDecision,ReportStatusChip} from '/src/routes/reports/report-view';

type Row=InboxRowOf<'report'>|InboxRowOf<'review'>;
export function InboxReportReader({audit,project,row,context}:{audit:Audit;project:Project;row?:Row;context:DetailContext}){
 const api=usePublicAPI(),query=useAuditReport(audit,api);
 const request=row?.type==='review'?row.decision.review:undefined;
 const owner=useReportAcceptance(api,audit.auditId,query.data,request?.requestId??null);
 const acceptance={...owner,onDecided:(result)=>{owner.onDecided(result);context.onDecided(row?.key??'report:'+audit.auditId,decisionOutcome(result.decision).label,project.name+' report');}};
 return <DetailPane header={<>
  <DetailBar where={row?.section==='decide'?'Decide':'Ready'} label={request?'Report acceptance':'Report'} actions={<OpenLink to={reportPath(audit.auditId)+(request?'?review='+encodeURIComponent(request.requestId):'')}>Open report</OpenLink>} context={context} focus={context.focusKey===row?.key}/>
  <DetailHeader title={project.name+' report'} status={query.data?<ReportStatusChip status={query.data.status}/>:undefined} meta={<><Link to={checkPath(project.projectId,audit.auditId)}>{checkTypeLabel(audit)}</Link><span>Updated <RecordedTime value={audit.updatedAt}/></span></>}/>
 </>} footer={showsAcceptance(acceptance)?<div className="reports-decision-footer"><ReportAcceptanceDecision auditId={audit.auditId} acceptance={acceptance}/></div>:undefined}>
  <Recorded text={context.recorded}/>
  <ReportContent audit={audit} query={query} acceptance={acceptance}/>
 </DetailPane>;
}
export function InboxReportReview({row,context}:{row:InboxRowOf<'review'>;context:DetailContext}){
 if(row.decision.review.kind!=='report-acceptance')return <OtherReview row={row} context={context}/>;
 return <InboxReportReader key={row.decision.audit.auditId} audit={row.decision.audit} project={row.decision.project} row={row} context={context}/>;
}
export function InboxReadyReport({row,context}:{row:InboxRowOf<'report'>;context:DetailContext}){return <InboxReportReader key={row.report.audit.auditId} audit={row.report.audit} project={row.report.project} row={row} context={context}/>;}
