import assert from 'node:assert/strict';import {readFile,writeFile,mkdir} from 'node:fs/promises';import {createHash} from 'node:crypto';import {chromium,expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
const root=new URL('./',import.meta.url),review=new URL('review/',root);await mkdir(review,{recursive:true});
const preserved=JSON.parse(await readFile(new URL('preserved-artifacts.json',root),'utf8'));for(const [path,hash] of Object.entries(preserved.files))assert.equal(createHash('sha256').update(await readFile(new URL('../../../../../'+path,root))).digest('hex'),hash,'Protected artifact changed: '+path);
const browser=await chromium.launch(),page=await browser.newPage(),errors=[],network=[],checks=[],parity=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(new URL(r.url()).pathname.startsWith('/v1/'))network.push(r.url())});const base='http://127.0.0.1:17560/test-results/inbox-production/current.html';
async function clean(){assert.deepEqual(await page.evaluate(()=>window.__inboxViolations),[]);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);assert.equal(await page.evaluate(()=>window.__inboxRequests.some(r=>/\/reviews(?:\/|$)|\/operations\/snapshot/.test(r.path))),false);await expect(page.getByText('Runs and capacity',{exact:true})).toHaveCount(0);await expect(page.getByText(/Active test approval|Report acceptance|awaiting owner acceptance/)).toHaveCount(0);}
async function open(view='overview',width=1440,theme='dark',state='normal',version='new'){
 await page.setViewportSize({width,height:1000});await page.goto(base+'?'+new URLSearchParams({view,width:String(width),theme,state,version}));await expect(page.locator('#main-content')).toBeVisible();
 if(state==='empty')await expect(page.getByRole('heading',{name:'Your Inbox is clear',includeHidden:true})).toBeAttached();
 else if(['error','partial'].includes(state)){await expect(page.getByRole('heading',{name:"Some Inbox items couldn't be loaded",includeHidden:true})).toBeAttached();assert.equal(await page.evaluate(()=>window.__inboxRouter.state.location.search),'');}
 else if(view==='overview'&&width<=820){await expect(page.getByRole('link',{name:'Debug information exposed in responses',exact:true})).toBeVisible();assert.equal(await page.evaluate(()=>window.__inboxRouter.state.location.search),'');}
 else if(view==='report'){
  if(state==='report-error')await expect(page.locator('.reports-body')).toContainText('Synthetic report read failure');
  else if(state==='report-pending')await expect(page.locator('.reports-body')).toContainText('has not reached report generation');
  else if(state==='report-unavailable')await expect(page.locator('.reports-body')).toContainText('completed without a summary report');
  else await expect(page.locator('.reports-markdown')).toContainText('End of report');
 }
 else if(['overview','issue','sql'].includes(view)){
  if(state==='detail-error')await expect(page.locator('.ui-panes-detail')).toContainText('Synthetic issue read failure');
  else {await expect(page.locator('.issues-workspace-title')).toBeVisible();await expect(page.locator('.issues-summary-grid')).toBeVisible();if(view==='sql')await expect(page.locator('.issues-evidence-preview pre').first()).toContainText('const sql');else await expect(page.getByRole('heading',{name:'Captured HTTP exchange',exact:true})).toBeVisible();}
 }
 else if(view==='blocked')await expect(page.getByRole('button',{name:'Retry model connection',exact:true})).toBeVisible();
 else if(view==='running')await expect(page.locator('.inbox-progress-summary')).toBeVisible();
 else await expect(page.getByRole('heading',{name:'Primary result',exact:true})).toBeVisible();
 if(['overview','issue','sql'].includes(view)&&state==='normal'&&!(view==='overview'&&width<=820))await expect(page.locator('.issues-summary-grid')).not.toContainText('Loading text…');
 await clean();checks.push({view,width,theme,state,version});
}
for(const theme of ['light','dark','black'])for(const width of [1440,1100,390,320])for(const view of ['overview','sql','report','blocked','running']){
 await open(view,width,theme);if([1440,390].includes(width)&&['overview','sql','report'].includes(view))await page.screenshot({path:new URL(`${view}-${theme}-${width}.png`,review).pathname,fullPage:true});
 if(theme==='dark'&&[1440,390].includes(width)&&['sql','report'].includes(view)){
  const selector=view==='sql'?'.issues-summary-grid':'.reports-body',content=await page.locator(selector).innerText(),font=await page.locator(selector).evaluateAll(es=>es.map(e=>{const s=getComputedStyle(e);return [s.fontFamily,s.fontSize,s.lineHeight]}));
  await open(view,width,theme,'normal',view==='sql'?'issues':'reports');assert.equal(await page.locator(selector).innerText(),content);assert.deepEqual(await page.locator(selector).evaluateAll(es=>es.map(e=>{const s=getComputedStyle(e);return [s.fontFamily,s.fontSize,s.lineHeight]})),font);parity.push({view,width});
 }
}
for(const state of ['report-error','report-pending','report-unavailable'])for(const width of [1440,390])await open('report',width,'dark',state);
for(const state of ['empty','partial','error'])for(const width of [1440,390])await open('overview',width,'dark',state);
await open('sql',390,'black','detail-error');
// Existing real report owner, exact full downloads and retained file facts.
await open('report');const expected=await page.evaluate(()=>window.__reportDocuments.audit_done);
for(const [name,suffix] of [['Download summary','.md'],['Download JSON','.json']]){const wait=page.waitForEvent('download');await page.getByRole('button',{name,exact:true}).click();const file=await wait;assert.equal(file.suggestedFilename(),'audit_done-report'+suffix);const bytes=await readFile(await file.path(),'utf8');if(suffix==='.md')assert.equal(bytes,expected.summary);else assert.deepEqual(JSON.parse(bytes),expected.machine);}
await page.getByText('Technical details',{exact:true}).click();await expect(page.locator('.reports-body')).toContainText('report.md@r1');
// Queue-first mobile entry and explicit Back/resize never trigger a new selection.
await open('overview',390);await page.getByRole('link',{name:'Product search builds SQL from request text without a parameter placeholder',exact:true}).click();await expect(page.locator('.issues-evidence-preview pre').first()).toContainText('const sql');await page.locator('.ui-panes-back').click();await expect(page.locator('.ui-panes-list')).toBeVisible();assert.equal(await page.evaluate(()=>window.__inboxRouter.state.location.search),'');await page.setViewportSize({width:1440,height:1000});await expect(page.getByRole('heading',{name:'Debug information exposed in responses',exact:true})).toBeVisible();assert.equal(await page.evaluate(()=>window.__inboxRouter.state.location.search),'');
// Actual annotation owner refuses demo writes; no accepted state is fabricated.
await open('sql',390);await page.getByRole('button',{name:'Not an issue',exact:true}).click();await page.getByRole('textbox',{name:'Why',exact:true}).fill('Synthetic preview only.');await page.getByRole('button',{name:'Record decision',exact:true}).click();await expect(page.locator('.ui-panes-detail')).toContainText('no live mutation was sent');await page.getByRole('tab',{name:'History',exact:true}).click();await expect(page.locator('.ui-panes-detail')).toContainText('Independent sources');await page.keyboard.press('j');await expect(page.locator('.inbox-detail-where')).toContainText('Unblock');await page.keyboard.press('k');await expect(page.locator('.issues-workspace-title')).toBeVisible();await clean();
await open('report',390,'dark','report-unavailable');await expect(page.getByRole('link',{name:'View task results →',exact:true})).toHaveAttribute('href','/projects/project_identity/audits/audit_done');await expect(page.getByRole('button',{name:'Download summary',exact:true})).toHaveCount(0);await page.screenshot({path:new URL('completed-without-report-390.png',review).pathname,fullPage:true});
assert.deepEqual(errors,[]);assert.deepEqual(network,[]);const report={layouts:checks.length,checks,contentParity:parity,preservedFiles:Object.keys(preserved.files).length,browserErrors:errors,liveAPIRequests:network,interactions:['desktop first selection','mobile queue/Back/resize','no approval or capacity requests/UI','complete issue evidence/snippets/history','full Reports parity and exact downloads','truthful completed-without-report task-result link','queue J/K and refused demo annotation']};await writeFile(new URL('verification.json',root),JSON.stringify(report,null,2));await browser.close();console.log(JSON.stringify({layouts:checks.length,parityCases:parity.length,preservedFiles:report.preservedFiles,browserErrors:errors,liveAPIRequests:network}));
