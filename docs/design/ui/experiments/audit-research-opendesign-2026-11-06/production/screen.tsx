// Review entry only. It renders the production router, never a duplicate UI.
import {createRoot} from 'react-dom/client';
import {createMemoryRouter} from 'react-router';
import {Application} from '/src/app/application';
import {applicationRoutes} from '/src/app/router';
import {initializeTheme} from '/src/app/theme';
import {PublicAPI} from '/src/api/client';
import {RunEventsManager} from '/src/events/run-events';
import {researchFixture,RESEARCH_PROJECT,RESEARCH_AUDIT} from '/src/test/audit-research-fixtures';
import {json,failure,emptyPageOrNotFound} from '/src/test/responses';
import sharedConfig from './shared-config.json';
import packageMetadata from '/package.json';
import '/src/app/theme.css';import '/src/styles.css';import '/src/app/ui-primitives.css';import '/src/ui/ui.css';
const params=new URLSearchParams(location.search),fixture=researchFixture(params.get('state')??'normal');
localStorage.setItem('contractor.theme',params.get('theme')??'dark');initializeTheme();
const api=new PublicAPI({uiVersion:packageMetadata.version,supportedApiVersions:['contractor.public.v1'],apiBaseUrl:location.origin},async input=>{
 const request=input instanceof Request?input:new Request(input),url=new URL(request.url);
 if(request.method!=='GET')return failure(403,'forbidden','Review fixture: writes are disabled. No live mutation was sent.');
 if(url.pathname==='/v1/config')return json(sharedConfig);
 if(url.pathname==='/v1/auth/session')return json({principal:{userId:'user_local',username:'demo',capabilities:['user']},csrfToken:'a'.repeat(43),idleExpiresAt:'2099-01-01T00:00:00Z',absoluteExpiresAt:'2099-01-02T00:00:00Z'});
 return await fixture.respond(request)??emptyPageOrNotFound(request,url);
});
class DemoSocket{protocol='contractor.events.v1';readyState=0;onopen=null;onmessage=null;onerror=null;onclose=null;constructor(){queueMicrotask(()=>{this.readyState=1;this.onopen?.(new Event('open'))})}send(value){const f=JSON.parse(value);if(f.type==='subscribe')queueMicrotask(()=>this.onmessage?.({data:JSON.stringify({version:'contractor.events.v1',type:'subscribed',subscriptionId:f.subscriptionId,stream:f.stream,cursor:f.after})}))}close(code=1000){this.readyState=3;this.onclose?.({code})}}
const section=params.get('tab')??'overview';
const router=createMemoryRouter(applicationRoutes(),{initialEntries:[`/projects/${RESEARCH_PROJECT}/audits/${RESEARCH_AUDIT}`+(section==='overview'?'':'/'+section)]});
window.__researchRouter=router;
const events=new RunEventsManager(location.origin,{WebSocketImplementation:DemoSocket});
createRoot(document.getElementById('root')!).render(<Application api={api} publicAPI={api} runEvents={events} router={router}/>);
