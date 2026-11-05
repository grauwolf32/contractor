import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';import {posix} from 'node:path';import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),repo=new URL('../../../../../',root),previous=new URL('../inbox-opendesign-2026-11-04/',root),target=new URL('ui/test-results/inbox-action-entry/',repo);await mkdir(target,{recursive:true});
await import(new URL('prepare.mjs',previous));
const resolveImports=source=>source.replace(/(from\s+|import\s+)(["'])(\.[^"']+)\2/g,(_,prefix,quote,path)=>prefix+quote+posix.normalize('/src/routes/inbox/'+path)+quote);
let route=resolveImports(await readFile(new URL('ui/src/routes/inbox/index.tsx',repo),'utf8')).replace('from "/src/routes/inbox/detail"','from "./focus-detail"');
const hook=`// Preview-only entry proposal: consume once, respecting explicit selection and mobile Back.
  const initialEntry = useRef(false);
  useLayoutEffect(() => {
    if (initialEntry.current) return;
    if (params.has("item") || window.matchMedia("(max-width: 820px)").matches) { initialEntry.current = true; return; }
    if (Object.values(data.pending).some(Boolean)) return;
    initialEntry.current = true;
    if (data.incomplete) return;
    const first = model.order[0];
    if (first !== undefined) move(first, { replace: true });
  }, [params, data.pending, data.incomplete, model.order, move]);
`;
route=route.replace('  const decideCount =',hook+'\n  const decideCount =');await writeFile(new URL('focus-route.tsx',target),route);
let detail=resolveImports(await readFile(new URL('ui/src/routes/inbox/detail.tsx',repo),'utf8')).replace('import { Overview, type DetailContext } from "/src/routes/inbox/detail-parts";','import { type DetailContext } from "/src/routes/inbox/detail-parts";\nimport { FocusOverview as Overview } from "./focus-overview";');await writeFile(new URL('focus-detail.tsx',target),detail);
for(const name of ['focus-overview.tsx','focus.css','index.html'])await copyFile(new URL(name,root),new URL(name,target));
let fixture=await readFile(new URL('current-fixture.tsx',previous),'utf8');
fixture=fixture.replace("import {createRoot}","import {InboxRoute as FocusInboxRoute} from './focus-route';\nimport './focus.css';\nimport {createRoot}");
const scenarios=`
if(['no-decisions','results-only','running-only'].includes(scenario)){server.findings={};server.reviews={};}
if(scenario==='results-only'){for(const key of Object.keys(server.audits))server.audits[key]=server.audits[key].filter(a=>a.state==='completed');server.runs=server.runs.filter(r=>r.state==='succeeded');}
if(scenario==='running-only'){for(const key of Object.keys(server.audits))server.audits[key]=server.audits[key].filter(a=>a.state==='active');server.runs=server.runs.filter(r=>r.state==='running');const active=server.audits.project_shop[0];server.workspaces.audit_trace=workspace(active,{gaps:0,unchecked:4});}
`;
fixture=fixture.replace('const routes=inboxRoutes(server);',scenarios+'\nconst routes=inboxRoutes(server);');
const customRoutes=`function previewRoutes(){const routes=applicationRoutes();if(!['current','previous'].includes(params.get('version')??'')){const index=routes[1].children[0].children[0];delete index.lazy;index.element=<FocusInboxRoute/>;}return routes;}\n`;
fixture=fixture.replace("createRoot(document.getElementById('root')!).render(",customRoutes+"createRoot(document.getElementById('root')!).render(").replace('createMemoryRouter(applicationRoutes(),','createMemoryRouter(previewRoutes(),');
const revision=createHash('sha256').update(fixture).update(route).update(detail).update(await readFile(new URL('focus-overview.tsx',root))).update(await readFile(new URL('focus.css',root))).digest('hex').slice(0,12);
await writeFile(new URL('focus-route.tsx',target),route.replace('./focus-detail"','./focus-detail?rev='+revision+'"'));await writeFile(new URL('focus-detail.tsx',target),detail.replace('./focus-overview"','./focus-overview?rev='+revision+'"'));
fixture=fixture.replace("'./focus-route'","'./focus-route?rev="+revision+"'").replace("'./focus.css'","'./focus.css?rev="+revision+"'").replaceAll('REVISION',revision).replaceAll('/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor',repo.pathname.replace(/\/$/,''));
await writeFile(new URL('current.tsx',target),fixture);await writeFile(new URL('current.html',target),'<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inbox · action entry</title><div id="root"></div><script type="module" src="./current.tsx?rev='+revision+'"></script></html>');
console.log('Prepared revised Inbox entry with real components; previous preview retained');
