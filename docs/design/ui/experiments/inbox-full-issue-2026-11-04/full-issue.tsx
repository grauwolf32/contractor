// Context adapter only: the complete issue renderer and decision owner are shared.
import {useSearchParams} from 'react-router';
import {IssueDetail} from '/src/routes/issues/issue-detail';
import {verdictLabel} from '/src/app/vocabulary';
import {DetailBar,OpenLink,Recorded,type DetailContext} from '/src/routes/inbox/detail-parts';
import {issuePath} from '/src/routes/inbox/present';
import type {InboxRowOf} from '/src/routes/inbox/model';

export function InboxFullIssue({row,context}:{row:InboxRowOf<'issue'>;context:DetailContext}){
 const [params]=useSearchParams(),selected=row.issue.finding;
 const issues=context.data.model.decide.filter(candidate=>candidate.type==='issue');
 const position=issues.findIndex(candidate=>candidate.key===row.key);
 const tab=params.get('tab'),search=tab==='evidence'||tab==='history'?'?tab='+tab:'';
 return <div className="issues-workspace inbox-full-issue">
  <Recorded text={context.recorded}/>
  <DetailBar where="Decide" label={`Possible issue ${position+1} of ${issues.length}`} actions={<><OpenLink to={`/projects/${encodeURIComponent(selected.projectId)}/findings`}>{row.issue.project.name}</OpenLink><OpenLink to={issuePath(selected.projectId,selected.findingId)+search}>Open in Issues</OpenLink></>} context={context} focus={context.focusKey===row.key}/>
  <IssueDetail key={row.key} projectId={selected.projectId} findingId={selected.findingId} listed={row.issue} projects={[row.issue.project]} position={position<0?undefined:{index:position,count:issues.length}} onDecided={result=>context.onDecided(row.key,verdictLabel(result.finding.analystVerdict).label,selected.title)}/>
 </div>;
}
