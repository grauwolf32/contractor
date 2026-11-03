import {chromium,expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {readFile,mkdir,writeFile} from 'node:fs/promises';import {createHash} from 'node:crypto';import {Script} from 'node:vm';import {gunzipSync} from 'node:zlib';
const root=new URL('./',import.meta.url),output=new URL('../../../../../ui/test-results/operations-design/verified/',root);
const read=name=>readFile(new URL(name,root),'utf8'),sha=value=>createHash('sha256').update(value).digest('hex');
const [seed,original,candidate,css,fixtures,provenance]=await Promise.all([read('seed.html'),read('original.html'),read('preview.html'),read('opendesign.css'),read('fixtures.json').then(JSON.parse),read('provenance.json').then(JSON.parse)]);
const split=seed.indexOf('</head>'),prefix=seed.slice(0,split),suffix=seed.slice(split);
expect(original.startsWith(prefix)&&original.endsWith(suffix)).toBe(true);
const inserted=original.slice(prefix.length,original.length-suffix.length).match(/^\s*<style\s+id=["']opendesign["']>\s*([\s\S]*?)\s*<\/style>\s*$/);
expect(inserted?.[1].trim()).toBe(css.trim());expect(candidate).toBe(original);
expect(css).not.toMatch(/@import|@font-face|url\s*\(|rgba?\s*\(|#[\da-f]{3,8}(?![\w-])|<\s*\/?\s*(?:script|style|iframe|object|link)|--font-(?:sans|mono)\s*:/i);
const declared=new Set([...seed.matchAll(/(--[\w-]+)\s*:/g),...css.matchAll(/(--[\w-]+)\s*:/g)].map(match=>match[1]));
const unknown=[...css.matchAll(/var\((--[\w-]+)/g)].map(match=>match[1]).filter(name=>!declared.has(name));expect(unknown).toEqual([]);
const scripts=html=>[...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map(match=>match[1]);expect(scripts(candidate)).toEqual(scripts(seed));new Script(scripts(candidate).at(-1));
expect(provenance.inputSha256).toBe(sha(seed));expect(provenance.status).toBe('succeeded');expect(provenance.cleanup.stopped&&provenance.cleanup.upstreamCredentialRemoved).toBe(true);expect(provenance.cleanup.requestsUsed).toBeLessThanOrEqual(24);expect(provenance.preservation.priorFilesMissing).toBe(0);expect(provenance.preservation.priorFilesChanged).toBe(0);expect(Object.values(provenance.preservation.tables).every(table=>table.allPriorRowsIdentical)).toBe(true);
const baselines=JSON.parse(gunzipSync(await readFile(new URL('baseline.json.gz',root))));expect(Object.keys(baselines)).toHaveLength(10);
for(const {snapshot}of Object.values(fixtures.fixtures)){if(snapshot){expect(snapshot.runtimeAgents.length).toBeLessThanOrEqual(8);expect(snapshot.allocations.length).toBeLessThanOrEqual(6)}}
await mkdir(output,{recursive:true});const browser=await chromium.launch(),errors=[],outbound=[],metrics=[];
const origin=process.env.OPERATIONS_PREVIEW_ORIGIN??'http://127.0.0.1:17568';
const expected=fixture=>{const snap=fixture.snapshot;return {counts:snap?Object.fromEntries(['idle','busy','reserved','draining','fenced'].map(state=>[state,snap.runtimeAgents.filter(agent=>agent.slotState===state).length])):null,mismatches:snap?snap.runtimeAgents.filter(agent=>agent.reconciliationReason!==undefined||agent.currentAllocationId!==agent.authoritativeAllocationId).length:null,processes:snap?.runtimeAgents.length??null,allocations:snap?.allocations.length??null,connection:fixture.connection}};
async function open({theme='dark',width=1440,scenario='attention',version='prototype'}={}){
 const page=await browser.newPage({viewport:{width,height:1000}});page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>{if(!request.url().startsWith(origin)&&!request.url().startsWith('data:'))outbound.push(request.url())});await page.goto(`${origin}/${version}.html?theme=${theme}&width=${width}&case=${scenario}`);await expect(page.getByRole('heading',{name:'Operations',exact:true})).toBeVisible();return page;
}
try{
 for(const theme of ['light','dark','black'])for(const width of [1440,1100,390,320])for(const scenario of ['normal','attention','empty','reconnecting','error']){
  const page=await open({theme,width,scenario});const fixture=fixtures.fixtures[scenario],summary=expected(fixture);
  expect(await page.evaluate(()=>window.__operationsReview.summary())).toEqual(summary);expect(await page.evaluate(()=>window.__operationsReview.snapshot())).toEqual(fixture.snapshot);
  for(const [id,value]of [['process-count',summary.processes],['allocation-count',summary.allocations],['mismatch-count',summary.mismatches]])await expect(page.locator('#'+id)).toHaveText(value===null?'—':String(value));
  await expect(page.locator('#connection')).toBeVisible();await expect(page.getByRole('button',{name:'Refresh',exact:true})).toBeVisible();
  if(scenario==='error'){await expect(page.locator('#read-error')).toBeVisible();await expect(page.locator('#snapshot-content')).toBeHidden();await page.getByRole('button',{name:'Try again',exact:true}).click();await expect(page.locator('#action-status')).toContainText('Synthetic read failure remains');}
  else{
   await expect(page.locator('#snapshot-content')).toBeVisible();await expect(page.locator('.capacity-note')).toBeVisible();await expect(page.locator('.agent-row')).toHaveCount(fixture.snapshot.runtimeAgents.length);await expect(page.locator('.allocation-row')).toHaveCount(fixture.snapshot.allocations.length);await expect(page.locator('#slot-legend li')).toHaveCount(5);
   for(const agent of fixture.snapshot.runtimeAgents)await expect(page.locator('#agent-'+agent.instanceId+' > summary')).toContainText(agent.slotState);
   if(scenario==='empty'){await expect(page.locator('#empty-runtime')).toBeVisible();await expect(page.locator('#slot-track')).toBeHidden();}
   if(scenario==='reconnecting'){await expect(page.locator('#connection-note')).toContainText('Showing the last snapshot');await expect(page.locator('#connection')).toHaveAttribute('data-tone','warning');}
  }
  const measured=await page.evaluate(()=>({width:innerWidth,content:document.documentElement.scrollWidth,font:getComputedStyle(document.body).fontFamily,overflow:[...document.querySelectorAll('main *')].filter(el=>{const r=el.getBoundingClientRect();return r.width>0&&(r.right>innerWidth+1||r.left< -1)}).slice(0,6).map(el=>el.className)}));expect(measured.content).toBeLessThanOrEqual(width);expect(measured.overflow).toEqual([]);metrics.push({theme,width,scenario,...measured});
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:new URL(`${scenario}-${theme}-${width}.png`,output).pathname,fullPage:true});await page.close();
 }
 for(const width of [1440,390,320]){
  const page=await open({width});const sidebar=page.locator('#sections-browser');if(width<=820)await sidebar.locator('summary').click();await expect(sidebar.locator('nav a')).toHaveCount(8);for(const link of await sidebar.locator('nav a').all())await expect(link).toBeVisible();
  await sidebar.getByRole('link',{name:'Settings',exact:true}).click();await expect(page.locator('#action-status')).toContainText('/operations/settings');if(width<=820)await sidebar.locator('summary').click();
  const reserved=page.locator('#agent-runtime-04');await reserved.locator('summary').click();await expect(reserved.locator('.row-details')).toContainText('Not reported');await expect(reserved.locator('.fact-list')).toContainText('Observed processidle');await expect(reserved.locator('.fact-list')).toContainText('Reconciled slotreserved');
  const fenced=page.locator('#agent-runtime-06');await fenced.locator('summary').click();await expect(fenced.locator('.row-details')).toContainText('alloc-observed-06');await expect(fenced.locator('.row-details')).toContainText('Authoritative allocationalloc-06');await expect(fenced.locator('.row-details')).toContainText('lease_confirmation_lost');await expect(fenced.locator('.row-details')).toContainText('likec4@1: read_likec4, validate_likec4');await expect(fenced.getByRole('button',{name:'Copy process ID runtime-06'})).toBeVisible();
  await fenced.getByRole('link',{name:'Inspect assigned allocation →'}).click();const allocation=page.locator('#allocation-alloc-06');await expect(allocation).toHaveAttribute('open','');await expect(allocation.locator('.fact-list')).toContainText('Authoritative phaseactive');await expect(allocation.locator('.fact-list')).toContainText('Observed phasefenced');
  await allocation.getByRole('link',{name:'Open Run →'}).click();await expect(page.locator('#action-status')).toContainText('/runs/run-analysis-06');
  await page.locator('.diagnostics > summary').click();await expect(page.locator('#snapshot-diagnostics-body')).toContainText('operations-generation-1');await expect(page.locator('#snapshot-diagnostics-body')).toContainText('Snapshot revision43');
  await page.getByRole('button',{name:'Refresh',exact:true}).click();await expect(page.locator('#action-status')).toContainText('Preview refreshed the same synthetic snapshot');
  if(width<=820){await page.getByRole('button',{name:'Menu',exact:true}).click();await expect(page.locator('#global-rail')).toBeVisible();await page.getByRole('button',{name:'Menu',exact:true}).click();await expect(page.locator('#global-rail')).toBeHidden();}
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);await page.close();
 }
 const live=await open({width:1440});await live.setViewportSize({width:390,height:1000});await expect(live.locator('#sections-browser')).not.toHaveAttribute('open','');expect(await live.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);await live.setViewportSize({width:1440,height:1000});await expect(live.locator('#sections-browser')).toHaveAttribute('open','');await live.close();
 expect(errors).toEqual([]);expect(outbound).toEqual([]);await writeFile(new URL('metrics.json',output),JSON.stringify(metrics,null,2)+'\n');console.log(`Passed ${metrics.length} native OpenDesign layouts, desktop/mobile navigation and drill-downs, exact fixtures/counts/unknown states, live resize, shared themes/fonts, no overflow/errors/external requests. Native CSS/HTML/controller provenance and preservation verified.`);
}finally{await browser.close()}
