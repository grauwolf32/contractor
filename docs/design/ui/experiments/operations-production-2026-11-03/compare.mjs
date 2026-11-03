import {chromium, expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {mkdir, writeFile, copyFile} from 'node:fs/promises';
import './prepare.mjs';
const root = new URL('./', import.meta.url), repo = new URL('../../../../../', root);
const retained = new URL('review/comparison/', root), served = new URL('ui/test-results/operations-production/comparison/', repo);
await mkdir(retained, {recursive:true}); await mkdir(served, {recursive:true});
const browser = await chromium.launch(), errors = [], metrics = [];
const sections = ['overview','runtime-agents','allocations','completed','performance','configuration','llm','credentials','settings'];
const headings = {overview:'Execution slots','runtime-agents':'Runtime Agents',allocations:'Current allocations',completed:'Completed allocation resources',performance:'Server performance',configuration:'Runtime configuration',llm:'LLM configurations',credentials:'Managed LLM credentials',settings:'Settings'};
try {
  const page = await browser.newPage(); page.on('pageerror', error => errors.push(error.message));
  await page.clock.setFixedTime(new Date('2026-11-03T12:00:30Z'));
  for (const section of sections) for (const width of [1440,390]) for (const version of ['approved','working']) {
    await page.setViewportSize({width,height:1000});
    const url = version === 'approved' ? 'http://127.0.0.1:17570/prototype.html' : 'http://127.0.0.1:17560/test-results/operations-production/current.html';
    await page.goto(`${url}?section=${section}&case=attention&theme=dark`);
    if (version === 'working') {
      await expect(page.getByRole('heading',{name:headings[section],exact:true})).toBeVisible();
      await expect(page.locator('.ops-page-body')).not.toContainText(/Loading [^\n]*…/);
      await expect(page.locator('.notice-error')).toHaveCount(0);
      if (section === 'performance') await expect(page.locator('.ops-bar-chart')).toHaveCount(2);
    } else await expect(page.locator('.overview-content').first()).not.toBeEmpty();
    await page.evaluate(() => document.fonts.ready);
    const result = await page.evaluate(() => ({width:innerWidth,content:document.documentElement.scrollWidth,font:getComputedStyle(document.body).fontFamily}));
    if (result.content > width) throw Error('Comparison overflow: '+JSON.stringify({section,version,...result}));
    metrics.push({section,version,state:'collapsed',...result});
    const capture = async state => {
      const name = `${section}-${width}-${version}-${state}.png`;
      await page.screenshot({path:new URL(name,retained).pathname,fullPage:true});
      await copyFile(new URL(name,retained),new URL(name,served));
    };
    await capture('collapsed');
    if (section === 'runtime-agents') {
      if (version === 'working') {
        const card = page.getByRole('article',{name:'default',exact:true});
        await card.locator('summary').click();
        await expect(card.getByRole('list',{name:'Toolsets for runtime-01'})).toBeVisible();
        await expect(card.getByText('read_likec4',{exact:true})).toBeVisible();
        await expect(card.getByText('validate_likec4',{exact:true})).toBeVisible();
        if (await card.locator('details').count() !== 1) throw Error('Agent capabilities must not require another disclosure');
        const columns = await card.locator('.ops-agent-facts').evaluate(el => getComputedStyle(el).gridTemplateColumns.split(' ').length);
        if (columns !== (width === 1440 ? 2 : 1)) throw Error('Agent capability columns differ from the approved layout');
        const positions = await card.locator('.ops-agent-facts > div').evaluateAll(els => els.slice(0,2).map(el => el.getBoundingClientRect().left));
        if ((positions[0] !== positions[1]) !== (width === 1440)) throw Error('Agent facts must occupy two actual columns on desktop');
      } else await page.locator('.inventory-entry').first().locator('summary').click();
      await capture('expanded');
    }
    console.log('Compared '+section+' '+width+' '+version);
  }
  if(errors.length)throw Error(JSON.stringify(errors));
  await writeFile(new URL('metrics.json',retained),JSON.stringify({metrics,errors,notes:['The shared production header/search remains global.','The real 15-second history has 60 observed bars; the original static sketch has 16 demo bars.','Server words and actual API identities remain authoritative.']},null,2)+'\n');
  const html = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Operations · OpenDesign / working UI</title><style>body{margin:0;background:#0b0e13;color:#dde3ec;font:13px system-ui}header{padding:14px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#11161f}select,a{font:inherit;color:inherit}select{background:#1a2431;border:1px solid #394453;padding:6px;border-radius:5px}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;padding:14px}.pair img{width:100%;display:block;border:1px solid #344050}.pair h2{font-size:14px}a{margin-left:auto}@media(max-width:640px){.pair{grid-template-columns:1fr}}</style><header><strong>OpenDesign / Working UI</strong><label>Section <select id="section">${sections.map(s=>'<option>'+s+'</option>').join('')}</select></label><label>Width <select id="width"><option>1440</option><option>390</option></select></label><label>Card <select id="state"><option>collapsed</option><option>expanded</option></select></label><a id="open" target="_blank">Open working UI →</a></header><main class="pair"><section><h2>Approved OpenDesign</h2><img id="approved" alt="Approved OpenDesign"></section><section><h2>Working React UI</h2><img id="working" alt="Working React UI"></section></main><script>const q=new URLSearchParams(location.search),section=document.getElementById('section'),width=document.getElementById('width'),state=document.getElementById('state');for(const el of [section,width,state]){el.value=q.get(el.id)||el.options[0].value;el.onchange=render}function render(){state.disabled=section.value!=='runtime-agents';if(state.disabled)state.value='collapsed';for(const version of ['approved','working'])document.getElementById(version).src=section.value+'-'+width.value+'-'+version+'-'+state.value+'.png';document.getElementById('open').href='../index.html?version=new&section='+section.value+'&case=attention&theme=dark&width='+width.value;history.replaceState(null,'','?section='+section.value+'&width='+width.value+'&state='+state.value)}render()</script></html>`;
  await writeFile(new URL('index.html',served),html);
  await writeFile(new URL('comparison.html',root),html);
  console.log('Retained 40 paired desktop/mobile screenshots, including expanded Runtime toolsets.');
} finally {await browser.close();}
