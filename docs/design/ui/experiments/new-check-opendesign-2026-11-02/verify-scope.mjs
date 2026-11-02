import {chromium,expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {mkdir,writeFile} from 'node:fs/promises';
const base='http://127.0.0.1:17567',root=new URL('../../../../../ui/test-results/new-check-design/',import.meta.url),capture=new URL('scope-captures/',root);
await mkdir(capture,{recursive:true});const browser=await chromium.launch(),errors=[],requests=[],documents=[];
async function open({width=1440,theme='dark',type='openapi-operation-trace',scenario='ready'}={}){
 const page=await browser.newPage({viewport:{width,height:1000}});page.setDefaultTimeout(10000);
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(new URL(r.url()).origin!==base||r.method()!=='GET')requests.push(r.url())});
 await page.goto(`${base}/prototype.html?type=${type}&case=${scenario}&theme=${theme}`);
 await expect(page.locator('#selection-title')).toHaveText(type==='source-checklist'?'Custom checklist':type==='owasp-top10-2025-source-risk'?'OWASP Top 10 (2025) review':'API endpoint trace');return page;
}
try {
 let layouts=0;
 for(const theme of ['light','dark','black'])for(const width of [1440,390])for(const example of ['api','source','checklist','editor','engagement']){
  const type=example==='source'?'owasp-top10-2025-source-risk':['checklist','editor'].includes(example)?'source-checklist':'openapi-operation-trace';
  const page=await open({width,theme,type,scenario:example==='engagement'?'engagement':'ready'});
  if(example==='source'){await expect(page.locator('.scope')).toBeHidden();await expect(page.locator('#summary-scope')).toHaveText('Whole source archive')}
  if(example==='api'){await expect(page.locator('#scope-copy')).toHaveText('All 6 endpoints in the selected API spec.');await page.getByText('View endpoints',{exact:true}).click();await expect(page.locator('.endpoint-list li')).toHaveCount(6)}
  if(example==='checklist')await expect(page.locator('#summary-scope')).toHaveText('All 3 items');
  if(example==='editor'){
   await page.locator('#write-checklist').click();await expect(page.locator('#checklist-editor textarea')).toHaveCount(3);await expect(page.locator('#start')).toBeDisabled();
   await page.locator('#apply-checklist').click();await expect(page.locator('#start')).toBeEnabled();await expect(page.locator('#summary-scope')).toHaveText('All 3 items');
   documents.push(await page.evaluate(()=>window.__scopeReview.document()));
  }
  if(example==='engagement'){
   await expect(page.locator('.scope #engagement-scope')).toBeVisible();await expect(page.locator('#engagement-scope')).toContainText('/api/admin');await expect(page.locator('.materials')).not.toContainText('Engagement');await expect(page.locator('#start')).toBeDisabled();
  }
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
  if(example==='editor')await page.locator('#checklist-editor').scrollIntoViewIfNeeded();
  if(example==='engagement')await page.locator('#engagement-scope').scrollIntoViewIfNeeded();
  await page.screenshot({path:new URL(`${example}-${theme}-${width}.png`,capture).pathname});await page.close();layouts++;
 }
 const page=await open({type:'source-checklist'});await page.locator('#write-checklist').click();
 const initial=await page.evaluate(()=>window.__scopeReview.draft());await page.locator('#apply-checklist').click();
 await page.getByRole('textbox',{name:'Item 1',exact:true}).fill('Verify that every order read is restricted to its owner.');await expect(page.locator('#start')).toBeDisabled();
 await page.locator('#apply-checklist').click();const edited=await page.evaluate(()=>window.__scopeReview.document());documents.push(edited);
 expect(edited.items[0].key).toBe(initial.items[0].key);expect(edited.items[0].version).toBe('2');expect(edited.items[1].version).toBe('1');
 await page.getByRole('button',{name:'Remove item 2',exact:true}).click();await expect(page.locator('#summary-scope')).toHaveText('2 draft items');await expect(page.locator('#start')).toBeDisabled();await page.locator('#apply-checklist').click();
 const removed=await page.evaluate(()=>window.__scopeReview.document());documents.push(removed);expect(removed.items.map(item=>item.key)).toEqual([initial.items[0].key,initial.items[2].key]);
 await page.locator('#add-checklist-item').focus();await page.keyboard.press('Enter');await expect(page.getByRole('textbox',{name:'Item 3',exact:true})).toBeFocused();await expect(page.locator('#apply-checklist')).toBeDisabled();
 await page.getByRole('textbox',{name:'Item 3',exact:true}).fill('Verify logout invalidates the session.');await page.locator('#apply-checklist').click();documents.push(await page.evaluate(()=>window.__scopeReview.document()));
 await page.getByRole('textbox',{name:'Item 1',exact:true}).fill('я'.repeat(33000));await expect(page.locator('#checklist-feedback')).toContainText('text limit');await expect(page.locator('#apply-checklist')).toBeDisabled();await expect(page.locator('#start')).toBeDisabled();
 await page.getByRole('textbox',{name:'Item 1',exact:true}).fill('Verify that every order read is restricted to its owner.');
 await page.getByRole('button',{name:'Use a project material',exact:true}).click();await expect(page.locator('#summary-scope')).toHaveText('All 3 items');await expect(page.locator('#start')).toBeEnabled();
 await page.locator('#write-checklist').click();await page.locator('#apply-checklist').click();
 await page.evaluate(()=>{
  while(document.querySelectorAll('#checklist-editor textarea').length<128){
   document.getElementById('add-checklist-item').click();const fields=document.querySelectorAll('#checklist-editor textarea'),last=fields[fields.length-1];last.value='Verify bounded checklist entry '+fields.length+'.';last.dispatchEvent(new Event('input',{bubbles:true}));
  }
 });
 await expect(page.locator('#add-checklist-item')).toBeDisabled();await expect(page.locator('#checklist-editor textarea')).toHaveCount(128);await page.locator('#apply-checklist').click();await expect(page.locator('#summary-scope')).toHaveText('All 128 items');documents.push(await page.evaluate(()=>window.__scopeReview.document()));
 await page.close();
 const empty=await open({width:390,type:'source-checklist'});await empty.locator('#write-checklist').click();
 for(let i=0;i<3;i++)await empty.getByRole('button',{name:'Remove item 1',exact:true}).click();
 await expect(empty.locator('#checklist-feedback')).toContainText('at least one item');await expect(empty.locator('#apply-checklist')).toBeDisabled();await expect(empty.locator('#start')).toBeDisabled();await empty.close();
 expect(errors).toEqual([]);expect(requests).toEqual([]);await writeFile(new URL('scope-generated.json',root),JSON.stringify(documents,null,2)+'\n');
 console.log(`Passed ${layouts} scope/editor/Engagement layouts across themes and desktop/phone, local editing, stable identities/versions, empty/dirty/UTF-8 limits, 128 items and no mutations/outbound requests.`);
} finally {await browser.close()}
