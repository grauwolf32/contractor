import {readFile,writeFile} from 'node:fs/promises';import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),prior=new URL('../operations-tabs-opendesign-2026-11-03/',root),repo=new URL('../../../../../',root);
try{await readFile(new URL('provenance.json',root));throw Error('Settings input is frozen; create a new design revision')}catch(e){if(e.code!=='ENOENT')throw e}
const [old,base,controller,template,theme,icons]=await Promise.all([readFile(new URL('preview.html',prior),'utf8'),readFile(new URL('base.css',root),'utf8'),readFile(new URL('controller.js',root),'utf8'),readFile(new URL('settings.html',root),'utf8'),readFile(new URL('ui/src/app/theme.css',repo),'utf8'),readFile(new URL('fixtures.json',prior),'utf8').then(s=>JSON.parse(s).icons)]);
const body=template.replace(/\{\{icon:([^}]+)\}\}/g,(_,name)=>icons[name]);
let shell=old.slice(0,old.indexOf('<div class="overview-content tabs-content"'));
if(!shell.includes('</head>'))throw Error('Reviewed shell not found');
shell=shell.replace('<title>Tabs · Contractor · Operations overview</title>','<title>Contractor · Settings design</title>').replace('</head>','<style id="settings-base">'+base+'</style></head>');
const hostSeed=shell+'<div class="overview-content">'+body+'</div></div><p id="action-status" class="action-status" role="status" aria-live="polite"></p></main></div></div><script>'+controller+'</script></body></html>';
// A smaller exact reference keeps model work bounded: same settings DOM, theme
// tokens and base rules, without the other tabs, shell or local controller.
const reference='<!doctype html><html data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'+theme+'\n*{box-sizing:border-box}body{margin:0;padding:24px;background:var(--background);color:var(--text);font:var(--text-sm)/1.5 var(--font-sans)}h2,h3,p{margin:0}h2{font-size:var(--text-lg)}button{font:inherit;color:inherit;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-sm);padding:var(--space-2) var(--space-3)}a{color:var(--link)}svg{width:18px;height:18px;flex:none;vertical-align:middle}'+base+'</style></head><body>'+body+'</body></html>';
await writeFile(new URL('seed.html',root),hostSeed);await writeFile(new URL('reference.html',root),reference);
const hash=s=>createHash('sha256').update(s).digest('hex');await writeFile(new URL('inputs.json',root),JSON.stringify({referenceSha256:hash(reference),hostSeedSha256:hash(hostSeed),priorTabsSha256:hash(old),priorOverviewSha256:hash(await readFile(new URL('../operations-opendesign-2026-11-02/preview.html',root),'utf8'))},null,2)+'\n');console.log('Built focused Settings reference and full-shell local seed.');
