import {chromium,expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),repo=new URL('../../../../../',root),port=process.env.OPERATIONS_TABS_PORT??17569;
const [seed,native,preview,css,provenance]=await Promise.all(['seed-submitted.html','assembled.html','preview.html','opendesign.css','provenance.json'].map(async file=>{const v=await readFile(new URL(file,root),'utf8');return file.endsWith('json')?JSON.parse(v):v}));
if(native!==preview)throw Error('Candidate differs from verified CSS-only assembly');
const inserted='<style id="opendesign-tabs">'+css+'</style>';
if(native!==seed.replace('</head>',inserted+'</head>')&&native!==seed.replace('</head>','<style id="opendesign-tabs">\n'+css+'\n</style></head>'))throw Error('Native output changed more than the CSS insertion');
if(createHash('sha256').update(seed).digest('hex')!==provenance.inputSha256)throw Error('Input hash mismatch');
if(provenance.status!=='native_css_complete_host_assembled'||provenance.nativeRunStatus!=='failed'||!provenance.cleanup.stopped||!provenance.cleanup.upstreamCredentialRemoved||provenance.preservation.priorFilesChanged||provenance.preservation.priorFilesMissing)throw Error('Missing honest native-output/assembly/preservation/cleanup proof');
for(const report of Object.values(provenance.preservation.tables))if(!report.allPriorRowsIdentical)throw Error('Prior history changed');
const baseTokens=new Set([...seed.matchAll(/(--[a-z0-9-]+)\s*:/g)].map(m=>m[1]));
for(const [,token]of css.matchAll(/var\((--[a-z0-9-]+)/g))if(!baseTokens.has(token))throw Error('Unknown native token '+token);
if(/#[0-9a-f]{3,8}\b|\brgba?\(|\bhsla?\(|@import|url\(/i.test(css))throw Error('Native CSS introduces raw color/remote resource');
if(createHash('sha256').update(css).digest('hex')!==provenance.cssSha256||createHash('sha256').update(native).digest('hex')!==provenance.htmlSha256)throw Error('Native CSS or assembled HTML hash mismatch');
const sections=['runtime-agents','allocations','completed','performance','configuration','llm','credentials','settings'];
const themes=['light','dark','black'],widths=[1440,1100,390,320],errors=[],external=[],metrics=[];
const out=new URL('ui/test-results/operations-tabs-design/verified/',repo);await mkdir(out,{recursive:true});await mkdir(new URL('review/',root),{recursive:true});
const browser=await chromium.launch();
function wire(page){page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:'+port)&&!r.url().startsWith('data:'))external.push(r.url())})}
try{
 const context=await browser.newContext();context.setDefaultTimeout(5000);
 for(const section of sections)for(const theme of themes)for(const width of widths){
  const page=await context.newPage();await page.setViewportSize({width,height:1000});wire(page);
  await page.goto(`http://127.0.0.1:${port}/prototype.html?section=${section}&theme=${theme}&width=${width}&case=attention`);
  await expect(page.locator('#tab-content h2')).toBeVisible();
  const m=await page.evaluate(()=>({viewport:innerWidth,documentWidth:document.documentElement.scrollWidth,font:getComputedStyle(document.body).fontFamily,heading:document.querySelector('#tab-content h2')?.textContent,icons:document.querySelectorAll('#tab-content svg').length}));
  if(m.documentWidth>width)throw Error('Overflow '+JSON.stringify({section,theme,width,...m}));
  if(!m.font.includes('system-ui'))throw Error('Global font mismatch');
  metrics.push({section,theme,width,...m});
  if(width===1440||width===390)await page.screenshot({path:new URL(`${section}-${theme}-${width}.png`,out).pathname,fullPage:true});
  if(theme==='dark'&&(width===1440||width===390))await page.screenshot({path:new URL(`review/${section}-${width}.png`,root).pathname,fullPage:true});
  await page.close();
 }
 for(const section of sections)for(const scenario of ['empty','error','reconnecting']){
  const page=await context.newPage();await page.setViewportSize({width:390,height:950});wire(page);await page.goto(`http://127.0.0.1:${port}/prototype.html?section=${section}&case=${scenario}&theme=black`);
  await expect(page.locator('#tab-content h2')).toBeVisible();if(scenario==='error')await expect(page.getByText('Read failed',{exact:true}).last()).toBeVisible();
  const size=await page.evaluate(()=>({w:innerWidth,d:document.documentElement.scrollWidth}));if(size.d>size.w)throw Error('Boundary overflow '+section+' '+scenario);
  await page.close();
 }
 const p=await context.newPage();wire(p);
 const open=async(section,width=390)=>{await p.setViewportSize({width,height:1000});await p.goto(`http://127.0.0.1:${port}/prototype.html?section=${section}&case=attention&theme=dark`)};
 await open('runtime-agents');await expect(p.locator('[data-agent]')).toHaveCount(7);await p.getByRole('button',{name:'Offline 1',exact:true}).click();await expect(p.locator('[data-agent]')).toHaveCount(1);await p.locator('[data-agent] summary').click();await p.getByRole('button',{name:'Edit labels',exact:true}).click();await expect(p.getByRole('dialog')).toBeVisible();await p.getByRole('button',{name:'Save labels',exact:true}).click();await expect(p.locator('.dialog-feedback')).toContainText('No Server changes');await p.getByRole('button',{name:'Close',exact:true}).click();
 await open('allocations');await p.locator('.inventory-entry summary').first().click();await expect(p.getByText('Control Plane: active',{exact:true}).first()).toBeVisible();await expect(p.getByText('Runtime: prepared',{exact:true}).first()).toBeVisible();await expect(p.getByRole('link',{name:'Open Run →',exact:true}).first()).toHaveAttribute('href','/runs/run-analysis-02');
 await open('completed');for(const status of ['available','partial','disabled','unavailable','unsupported','pending'])await expect(p.locator('.entry-states').getByText(status,{exact:true}).first()).toBeVisible();await p.getByLabel('Run ID',{exact:true}).fill('run-api-02');await p.getByRole('button',{name:'Apply filter',exact:true}).click();await expect(p.locator('.inventory-entry')).toHaveCount(1);await p.getByRole('button',{name:'Clear',exact:true}).click();await expect(p.locator('.inventory-entry')).toHaveCount(6);
 await open('performance');await p.getByLabel('Time range').selectOption('7d');await expect(p.getByText('No demo samples for this range',{exact:true})).toHaveCount(2);await expect(p.locator('.chart')).toHaveCount(0);
 await open('llm');await p.getByRole('button',{name:'Agent templates',exact:true}).click();await expect(p.getByText(/Agent templates are read-only here/)).toBeVisible();await expect(p.getByRole('button',{name:'Clone as new version',exact:true})).toHaveCount(0);await p.getByRole('button',{name:'LLM gateways',exact:true}).click();await p.locator('.inventory-entry summary').first().click();await p.getByRole('button',{name:'Clone as new version',exact:true}).click();await expect(p.getByRole('dialog')).toBeVisible();await p.getByRole('button',{name:'Publish new version',exact:true}).click();await expect(p.locator('.dialog-feedback')).toContainText('No Server changes');await p.getByRole('button',{name:'Close',exact:true}).click();
 for(const [action,title]of [['configuration','Publish RuntimeConfig'],['credentials','Create LLM credential']])for(const width of [1440,390,320]){await open(action,width);await p.getByRole('button',{name:title,exact:true}).click();await expect(p.getByRole('dialog')).toBeVisible();const d=await p.getByRole('dialog').evaluate(el=>({w:el.clientWidth,s:el.scrollWidth}));if(d.s>d.w)throw Error('Dialog overflow '+action+' '+width);await p.getByRole('button',{name:'Close',exact:true}).click()}
 await open('settings');await p.getByLabel('Concurrent Runs using a model').fill('3');await p.getByRole('button',{name:'Save scheduling limits',exact:true}).click();await expect(p.locator('#action-status')).toContainText('No Server changes');await p.getByLabel('Concurrent Runs using a model').fill('33');await p.getByRole('button',{name:'Save scheduling limits',exact:true}).click();if(await p.getByLabel('Concurrent Runs using a model').evaluate(e=>e.validity.valid))throw Error('Invalid scheduling limit accepted');await p.getByRole('button',{name:'Reset',exact:true}).first().click();await expect(p.getByLabel('Concurrent Runs using a model')).toHaveValue('1');
 // Wrapper navigation goes through all sections, including the unchanged overview.
 await p.setViewportSize({width:1600,height:1100});await p.goto(`http://127.0.0.1:${port}/?section=runtime-agents&theme=dark&width=1440`);
 const frame=()=>p.frameLocator('iframe');
 for(const section of ['allocations','performance','configuration','llm','credentials','settings','overview','runtime-agents']){const label=section==='llm'?'LLM configurations':section==='runtime-agents'?'Runtime Agents':section==='configuration'?'Configuration':section[0].toUpperCase()+section.slice(1);await frame().getByRole('navigation',{name:'Operations sections',exact:true}).getByRole('link',{name:label,exact:true}).click();await expect(p.locator('#section')).toHaveValue(section);await expect(frame().getByRole('navigation',{name:'Operations sections',exact:true}).getByRole('link',{name:label,exact:true})).toHaveAttribute('aria-current','page')}
 // Mobile section menu, keyboard-native disclosure, allocation deep target.
 await p.locator('#width').selectOption('390');await expect(frame().locator('#sections-browser')).not.toHaveAttribute('open');await frame().locator('#sections-browser>summary').focus();await p.keyboard.press('Enter');await expect(frame().locator('#sections-browser')).toHaveAttribute('open');await frame().getByRole('navigation',{name:'Operations sections',exact:true}).getByRole('link',{name:'Allocations',exact:true}).click();await expect(p.locator('#section')).toHaveValue('allocations');
 await p.locator('#width').selectOption('live');await p.setViewportSize({width:440,height:1000});await expect.poll(async()=>{try{return await p.frames()[1].evaluate(()=>innerWidth)}catch{return 0}}).toBe(408);
 if(errors.length||external.length)throw Error(JSON.stringify({errors,external}));
 await writeFile(new URL('review/metrics.json',root),JSON.stringify({layouts:metrics,boundaries:24,errors,external},null,2)+'\n');
 console.log(`Verified ${metrics.length} OpenDesign layouts, 24 boundary examples, filters, history states, diagnostics, dialogs, forms, section navigation and live resize; no overflow, page errors or external requests.`);
}finally{await browser.close()}
