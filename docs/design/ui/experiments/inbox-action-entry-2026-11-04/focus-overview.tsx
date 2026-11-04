import {Link} from 'react-router';
import {DetailPane,StatusGlyph,Button} from '/src/ui';
import type {DetailContext} from '/src/routes/inbox/detail-parts';
import {SECTIONS,checkName} from '/src/routes/inbox/present';
import type {InboxRow} from '/src/routes/inbox/model';

function title(row:InboxRow):string {
 if(row.type==='issue')return row.issue.finding.title;
 if(row.type==='run')return row.run.workflow;
 if(row.type==='check')return checkName(row.check);
 if(row.type==='report')return row.report.project.name+' report';
 return row.decision.project.name+' · pending review';
}
/** Compact unselected states; ordinary desktop entry opens the actual detail. */
export function FocusOverview({context}:{context:DetailContext;counts?:unknown}){
 const {data}=context,first=data.model.order[0],pending=Object.values(data.pending).some(Boolean);
 if(pending)return <DetailPane><div className="inbox-focus-state" role="status"><StatusGlyph tone="progress"/><h2>Loading your Inbox</h2><p>Reading decisions, stopped work and results.</p></div></DetailPane>;
 if(data.incomplete)return <DetailPane><div className="inbox-focus-state" role="status"><StatusGlyph tone="warning"/><h2>Some Inbox items couldn't be loaded</h2><p>Loaded items are still available in the queue. Retry to refresh the missing information.</p><Button onClick={()=>void data.retry()}>Retry</Button></div></DetailPane>;
 if(!first)return <DetailPane><div className="inbox-focus-state"><StatusGlyph tone="done"/><h2>Your Inbox is clear</h2><p>No decisions, stopped work, recent results or running checks are listed.</p><div className="inbox-focus-actions"><Link className="ui-btn" data-variant="primary" to="/checks/new">Start a check</Link><Link to="/projects">Browse projects</Link></div></div></DetailPane>;
 return <DetailPane><div className="inbox-focus-state"><p className="inbox-focus-eyebrow">{SECTIONS[first.section].title}</p><h2>{title(first)}</h2><p>Choose an item from the queue, or continue with this one.</p><Button variant="primary" onClick={()=>context.onSelect(first)}>Continue</Button></div></DetailPane>;
}
