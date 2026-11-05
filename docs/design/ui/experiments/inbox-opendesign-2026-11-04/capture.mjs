import {chromium} from '../../../../../ui/node_modules/@playwright/test/index.mjs';import {mkdir,writeFile,readFile} from 'node:fs/promises';
const root=new URL('./',import.meta.url);await mkdir(new URL('reference/',root),{recursive:true});
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:1000}}),report=[],fragments=[];
const errors=[];page.on('pageerror',e=>errors.push(e.message));
for(const view of ['overview','issue','review','blocked','ready','running']){
 await page.goto('http://127.0.0.1:17560/test-results/inbox-design/current.html?version=current&view='+view+'&theme=dark');await page.waitForTimeout(800);await page.locator('.ui-panes-detail').waitFor({state:'attached'});
 await page.screenshot({path:new URL('reference/current-'+view+'.png',root).pathname,fullPage:true});
 const data=await page.evaluate(()=>({body:document.body.innerText,violations:window.__inboxViolations,html:document.querySelector('.ui-panes')?.outerHTML??document.querySelector('main').innerHTML}));
 report.push({view,text:data.body,violations:data.violations});fragments.push('<!-- '+view+' -->\n'+(view==='overview'?data.html:await page.locator('.ui-panes-detail').innerHTML()).replace(/<svg[\s\S]*?<\/svg>/g,'<svg aria-hidden="true"><!-- existing icon --></svg>'));
}
const tokens=await readFile(new URL('../../../../../ui/src/app/theme.css',root),'utf8');
const css=await readFile(new URL('../../../../../ui/src/routes/inbox/inbox.css',root),'utf8');
await writeFile(new URL('reference/inbox-reference.html',root),'<!doctype html><html><head><style>'+tokens+'</style><style>'+css+'</style></head><body class="inbox-redesign">'+fragments.map((f,i)=>i===0?f:f.slice(f.indexOf('<section class="ui-detail')>=0?f.indexOf('<section class="ui-detail'):0)).join('\n')+'</body></html>',{flag:'wx'}).catch(error=>{if(error.code!=='EEXIST')throw error});
await writeFile(new URL('reference/baseline.json',root),JSON.stringify({errors,views:report},null,2));console.log(JSON.stringify({errors,views:report.map(x=>({view:x.view,violations:x.violations,text:x.text.slice(-1900)})),referenceBytes:(await readFile(new URL('reference/inbox-reference.html',root))).length}));await browser.close();
