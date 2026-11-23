// Design-only browser review. This does not replace the production/API gates.
import {chromium} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const root=new URL('./',import.meta.url);await mkdir(new URL('review/',root),{recursive:true});
const browser=await chromium.launch({headless:true});
const report={scope:'Synthetic OpenDesign review; not production or real API acceptance',matrix:[],interactions:[],errors:[]};
try {
  const page=await browser.newPage();
  page.on('pageerror',e=>report.errors.push(e.message));
  for(const width of [1440,1100,390,320])for(const theme of ['light','dark','black'])for(const tab of ['overview','hypotheses','issues','materials','work','activity']){
    await page.setViewportSize({width,height:1000});
    await page.goto(`http://127.0.0.1:17566/native.html?tab=${tab}&theme=${theme}`);
    await page.waitForFunction(()=>window.AUDIT_PREVIEW_READY===true);
    const visible=page.locator(`[data-panel="${tab}"]`);assert(await visible.isVisible());
    const bounds=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,width:innerWidth,font:getComputedStyle(document.body).fontFamily,missing:[]}));
    assert(bounds.scroll<=width,`${width}/${theme}/${tab}: overflow ${bounds.scroll}`);
    assert(bounds.font.includes('system-ui'),`Shared font missing: ${bounds.font}`);
    const bad=await page.locator('button:visible,input:visible,select:visible,textarea:visible').evaluateAll(nodes=>nodes.filter(n=>{const r=n.getBoundingClientRect();return r.left<-.5||r.right>innerWidth+.5}).map(n=>n.textContent?.slice(0,70)||n.getAttribute('aria-label')));
    // Horizontal section navigation intentionally scrolls as a single strip.
    const badOutsideTabs=await page.locator('button:visible,input:visible,select:visible,textarea:visible').evaluateAll(nodes=>nodes.filter(n=>{const r=n.getBoundingClientRect();return !n.closest('.section-tabs')&&(r.left<-.5||r.right>innerWidth+.5)}).map(n=>n.textContent?.slice(0,70)||n.getAttribute('aria-label')));
    assert.deepEqual(badOutsideTabs,[],`${width}/${theme}/${tab}: control outside viewport`);
    report.matrix.push({width,theme,tab,overflow:false,sharedFont:true,scrollingTabControls:bad.length});
    if((width===1440&&theme==='dark'&&['overview','hypotheses','materials'].includes(tab))||(width===390&&theme==='black'&&['overview','issues','hypotheses'].includes(tab)))await page.screenshot({path:new URL(`review/${tab}-${width}-${theme}.png`,root).pathname,fullPage:true});
  }
  await page.setViewportSize({width:390,height:920});
  await page.goto('http://127.0.0.1:17566/native.html?theme=black');
  await page.getByRole('button',{name:'Pause new Audit Runs',exact:true}).click();
  await page.getByText('New Audit Runs are paused.',{exact:false}).waitFor();
  assert(await page.getByRole('button',{name:'Resume accepted work'}).isVisible());report.interactions.push('Pause retains active Run and offers Resume');
  await page.getByRole('button',{name:'Resume accepted work'}).click();
  await page.getByRole('tab',{name:/Hypotheses/}).click();
  await page.getByRole('button',{name:/Generated Not checked A tenant/}).click();
  assert(await page.getByText('This hypothesis is registered. No checking task has been accepted for it.',{exact:true}).isVisible());
  report.interactions.push('Registered hypothesis is separate from scheduled check');
  await page.getByRole('button',{name:'Add hypothesis',exact:true}).click();
  await page.getByLabel('Statement',{exact:true}).fill('A customer can access another account’s invoice');
  await page.getByLabel('Scope & conditions',{exact:true}).fill('Synthetic baseline; two independent accounts');
  await page.getByLabel('Why investigate this',{exact:true}).fill('Selected order access observations');
  await page.getByRole('button',{name:'Save hypothesis',exact:true}).click();
  assert(await page.getByRole('heading',{name:'A customer can access another account’s invoice',exact:true}).isVisible());report.interactions.push('Manual hypothesis can save without scheduling a check');
  await page.getByRole('tab',{name:/Materials/}).click();
  await page.getByRole('button',{name:'Preview',exact:true}).first().click();
  assert(await page.getByRole('dialog').isVisible());
  await page.getByRole('dialog').getByRole('button',{name:'Publish to Project',exact:true}).click();
  assert(await page.getByRole('heading',{name:'Publish exact material to Project',exact:true}).isVisible());
  await page.getByRole('button',{name:'Publish exact revision',exact:true}).click();report.interactions.push('Exact material preview and explicit Project publication dialog');
  await page.getByRole('tab',{name:/Work/}).click();
  await page.getByRole('button',{name:'Add work',exact:true}).click();
  await page.getByLabel('What should this work investigate?',{exact:true}).fill('Explore invoice export access');
  await page.getByRole('button',{name:'Accept work',exact:true}).click();
  assert(await page.getByText('9 accepted tasks',{exact:true}).isVisible());
  assert(await page.getByText('Explore invoice export access',{exact:true}).isVisible());report.interactions.push('Ordinary work selects a material set and updates task totals');
  await page.goto('http://127.0.0.1:17566/native.html?theme=black&state=conflict');
  await page.waitForFunction(()=>window.AUDIT_PREVIEW_READY===true);
  const statement=page.getByLabel('Statement',{exact:true});await statement.fill('Unsaved analyst text');
  await page.getByRole('button',{name:'Save hypothesis',exact:true}).click();
  assert.equal(await statement.inputValue(),'Unsaved analyst text');assert(await page.getByRole('button',{name:'Save hypothesis',exact:true}).isDisabled());
  await page.getByRole('button',{name:'Reload current assertion',exact:true}).click();assert(await page.getByRole('button',{name:'Save hypothesis',exact:true}).isEnabled());report.interactions.push('Stale edit keeps draft and requires explicit reload');
  await page.goto('http://127.0.0.1:17566/native.html?theme=dark&state=stale-read');
  await page.waitForFunction(()=>window.AUDIT_PREVIEW_READY===true);
  assert(await page.getByText('Could not refresh this check.',{exact:true}).isVisible());assert(await page.getByRole('heading',{name:'Current work',exact:true}).isVisible());report.interactions.push('Failed refresh keeps loaded content');
  assert.deepEqual(report.errors,[]);
  // The original native file must retain the complete semantic interaction contract.
  const seed=await readFile(new URL('native-inputs/seed.html',root),'utf8'),native=await readFile(new URL('native/index.html',root),'utf8');
  const ids=[...seed.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
  report.nativeIDs={expected:ids.length,missing:ids.filter(id=>!native.includes(`id="${id}"`))};
  assert.deepEqual(report.nativeIDs.missing,[]);
  report.passed=true;
} catch(e) {report.passed=false;report.failure=String(e);throw e;}
finally {await browser.close();await writeFile(new URL('verification.json',root),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.passed,matrix:report.matrix.length,interactions:report.interactions.length,errors:report.errors,failure:report.failure}));}
