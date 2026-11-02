import { chromium, expect } from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root=new URL('../../../../../',import.meta.url);
const capture=new URL('ui/test-results/new-check-production/',root);
await mkdir(capture,{recursive:true});
let source=await readFile(new URL('./production-fixture.tsx',import.meta.url),'utf8');
source=source.replaceAll('/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor',root.pathname.replace(/\/$/,''));
await writeFile(new URL('screen.tsx',capture),source);
const rev=createHash('sha256').update(source).digest('hex').slice(0,12);
await writeFile(new URL('screen.html',capture),`<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Contractor · New check · production UI</title><div id="root"></div><script type="module" src="./screen.tsx?rev=${rev}"></script></html>`);
await writeFile(new URL('index.html',capture),`<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>New check · production preview</title><style>body{margin:0;background:#191c22;color:#fff;font:13px system-ui}header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px}select{font:inherit}iframe{display:block;border:0;margin:auto;height:calc(100dvh - 53px);max-width:100%}</style><header>Production UI · synthetic read-only data <select id="theme"><option>light</option><option>dark</option><option>black</option></select><select id="width"><option>1440</option><option>1100</option><option>390</option><option>320</option></select><select id="case"><option>ready</option><option>source</option><option>checklist</option><option>editor</option><option>asvs</option><option>engagement</option><option>missing</option><option>unsupported</option><option>error</option></select></header><iframe title="New check production UI"></iframe><script>const p=new URLSearchParams(location.search);for(const id of ['theme','width','case']){const s=document.getElementById(id);s.value=p.get(id)||(id==='theme'?'dark':id==='width'?'1440':'ready');s.onchange=()=>{p.set(id,s.value);location.search=p}}const f=document.querySelector('iframe');f.style.width=(p.get('width')||1440)+'px';f.src='./screen.html?'+p;</script></html>`);
const origin=process.env.NEW_CHECK_PREVIEW_ORIGIN??'http://127.0.0.1:17560';
const browser=await chromium.launch();const errors=[],outbound=[],metrics=[];
async function open(width,theme,example){
 const page=await browser.newPage({viewport:{width,height:1000}});
 page.on('pageerror',error=>errors.push(error.message));
 page.on('request',request=>{if(!request.url().startsWith(origin)&&!request.url().startsWith('data:'))outbound.push(request.url())});
 await page.goto(`${origin}/test-results/new-check-production/screen.html?theme=${theme}&case=${example}`);
 await expect(page.getByRole('heading',{name:'Start a check',exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'Start check',exact:true})).toBeVisible();
 return page;
}
try{
 for(const theme of ['light','dark','black'])for(const width of [1440,1100,390,320])for(const example of ['ready','source','checklist','editor','asvs','engagement','missing','unsupported','error']){
  if([1100,320].includes(width)&&example!=='editor')continue;
  const page=await open(width,theme,example);
  if(example==='ready'){await expect(page.locator('.start-scope-size').first()).toHaveText('6 endpoints in the API spec');await page.getByText('View endpoints',{exact:true}).click();await expect(page.locator('.start-scope-operations li')).toHaveCount(6);await expect(page.getByRole('button',{name:'Start check',exact:true})).toBeEnabled();}
  if(example==='source')await expect(page.locator('.start-scope-card')).toHaveCount(0);
  if(example==='checklist')await expect(page.locator('.start-summary-facts')).toContainText('3 items in the checklist');
  if(example==='editor'){
   await page.getByRole('button',{name:'Write checklist'}).click();await expect(page.locator('.start-checklist-editor textarea')).toHaveCount(1);await expect(page.getByLabel('Item 1',{exact:true})).toHaveValue('');await expect(page.getByRole('button',{name:'Start check',exact:true})).toBeDisabled();
   await page.locator('.start-checklist-editor').scrollIntoViewIfNeeded();await page.screenshot({path:new URL(`editor-empty-${theme}-${width}.png`,capture).pathname});
   await page.getByLabel('Item 1',{exact:true}).fill('Verify order reads are limited to the owner.');await page.getByRole('button',{name:'Use checklist',exact:true}).click();await expect(page.getByRole('button',{name:'Start check',exact:true})).toBeEnabled();await expect(page.locator('.start-summary-facts')).toContainText('1 item');
  }
  if(example==='asvs'){await expect(page.locator('.start-scope-card')).toHaveCount(0);await page.getByText('Advanced options',{exact:true}).click();await expect(page.getByRole('group',{name:'Review coverage'})).toBeVisible();}
  if(example==='engagement'){await expect(page.locator('.start-engagement')).toContainText('https://example.test/api');await expect(page.locator('.start-card').filter({has:page.getByRole('heading',{name:'Materials',exact:true})})).not.toContainText('Engagement');await expect(page.getByRole('button',{name:'Start check',exact:true})).toBeDisabled();}
  if(['missing','unsupported','error'].includes(example))await expect(page.getByRole('button',{name:'Start check',exact:true})).toBeDisabled();
  if(width<=820){expect(await page.locator('.start-summary-card').evaluate(el=>getComputedStyle(el).position)).toBe('static');await page.locator('.start-type-picker > summary').click();await expect(page.getByRole('searchbox',{name:'Find a check type'})).toBeVisible();await page.locator('.start-type-picker > summary').click();}
  const metric=await page.evaluate(()=>({scrollWidth:document.documentElement.scrollWidth,width:innerWidth,font:getComputedStyle(document.querySelector('.start-workspace')).fontFamily,overflow:[...document.querySelectorAll('.start-workspace *')].filter(el=>{const r=el.getBoundingClientRect();return r.width>0&&(r.right>innerWidth+1||r.left< -1)}).slice(0,5).map(el=>el.className)}));
  expect(metric.scrollWidth).toBeLessThanOrEqual(width);expect(metric.overflow).toEqual([]);metrics.push({theme,width,example,...metric});
  if(example==='editor')await page.locator('.start-checklist-editor').scrollIntoViewIfNeeded();else await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:new URL(`${example}-${theme}-${width}.png`,capture).pathname,fullPage:true});await page.close();
 }
 expect(errors).toEqual([]);expect(outbound).toEqual([]);await writeFile(new URL('metrics.json',capture),JSON.stringify(metrics,null,2)+'\n');console.log(`Passed ${metrics.length} production layouts: actual React route, scope/editor/Engagement, responsive picker, global fonts/themes, no page overflow/errors/external requests. Screenshots in ui/test-results/new-check-production.`);
}finally{await browser.close()}
