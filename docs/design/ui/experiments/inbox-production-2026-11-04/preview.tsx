// Actual production routes; all synthetic API requests are intercepted in memory.
import {createRoot} from 'react-dom/client';import {createMemoryRouter} from 'react-router';
import {Application} from '/src/app/application';import {applicationRoutes} from '/src/app/router';import {initializeTheme} from '/src/app/theme';import {PublicAPI} from '/src/api/client';import {RunEventsManager} from '/src/events/run-events';
import {json,failure,emptyPageOrNotFound,operationOf} from '/src/test/responses';import {responseViolation} from '/src/api/contract';
import {emptyServer,busyServer,inboxRoutes} from './fixtures?rev=REVISION';import {fullIssueFixtures} from './issue-fixtures?rev=REVISION';import {reportFixtures} from './report-fixtures?rev=REVISION';
import packageMetadata from '/package.json';import config from '/@fs//Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor/api/testdata/v1alpha1/valid/shared-config.json';
import '/src/app/theme.css';import '/src/styles.css';import '/src/app/ui-primitives.css';import '/src/ui/ui.css';
const params=new URLSearchParams(location.search),scenario=params.get('state')??'normal',view=params.get('view')??'overview';localStorage.setItem('contractor.theme',params.get('theme')??'dark');initializeTheme();
const server=scenario==='empty'?emptyServer():busyServer(),rich=fullIssueFixtures(server,scenario),reports=await reportFixtures(server,scenario),routes=inboxRoutes(server);window.__reportDocuments=reports.documents;window.__inboxRequests=[];window.__inboxViolations=[];
const api=new PublicAPI({uiVersion:packageMetadata.version,supportedApiVersions:['contractor.public.v1'],apiBaseUrl:location.origin},async input=>{
 const request=input instanceof Request?input:new Request(input),url=new URL(request.url),path=url.pathname;window.__inboxRequests.push({method:request.method,path});
 if(request.method!=='GET')return failure(403,'forbidden','Design preview: mutations are disabled; no live mutation was sent.');
 let response;
 if(path==='/v1/config')response=json(config);
 else if((scenario==='error'&&['/v1/audits','/v1/findings','/v1/runs'].includes(path))||(scenario==='partial'&&path==='/v1/findings'))response=failure(503,'storage_transaction_conflict','Synthetic Inbox read failure');
 else response=reports.response(request,url)??rich.response(request,url);
 if(!response)for(const [pattern,handler] of Object.entries(routes)){
  const [method,template]=pattern.split(' ');if(method!==request.method)continue;const names=[];const match=path.match(new RegExp('^'+template.replace(/\{([^}]+)\}/g,(_,name)=>{names.push(name);return '([^/]+)'})+'$'));if(!match)continue;response=await handler({request,url,params:Object.fromEntries(names.map((name,i)=>[name,decodeURIComponent(match[i+1])]))});break;
 }
 response??=emptyPageOrNotFound(request,url);const operation=operationOf(request.method,path);if(operation&&response.headers.get('Content-Type')?.includes('json')){const violation=responseViolation(operation.operation,response.status,await response.clone().json());if(violation)window.__inboxViolations.push({path,violation});}return response;
});
class DemoSocket{protocol='contractor.events.v1';readyState=0;onopen=null;onmessage=null;onerror=null;onclose=null;constructor(){queueMicrotask(()=>{this.readyState=1;this.onopen?.(new Event('open'))})}send(value){const f=JSON.parse(value);if(f.type==='subscribe')queueMicrotask(()=>this.onmessage?.({data:JSON.stringify({version:'contractor.events.v1',type:'subscribed',subscriptionId:f.subscriptionId,stream:f.stream,cursor:f.after})}))}close(code=1000){this.readyState=3;this.onclose?.({code})}}
const views={issue:'issue:project_shop:finding_idor',sql:'issue:project_shop:finding_sqli',report:'report:audit_done',blocked:'run:run_waiting',ready:'run:run_ok',running:'check:audit_trace'},tab=params.get('tab')??'summary';
const entry=params.get('version')==='issues'?'/issues/project_shop/'+(view==='sql'?'finding_sqli':'finding_idor')+'?state=all&tab='+tab:params.get('version')==='reports'?'/reports/audit_done':views[view]?'/?item='+encodeURIComponent(views[view])+(tab==='summary'?'':'&tab='+tab):'/';
const router=createMemoryRouter(applicationRoutes(),{initialEntries:[entry]});window.__inboxRouter=router;const events=new RunEventsManager(location.origin,{WebSocketImplementation:DemoSocket});createRoot(document.getElementById('root')!).render(<Application api={api} publicAPI={api} runEvents={events} router={router}/>);
