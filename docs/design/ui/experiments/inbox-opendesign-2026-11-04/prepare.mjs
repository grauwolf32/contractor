import {mkdir,readFile,writeFile,copyFile} from 'node:fs/promises';import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),repo=new URL('../../../../../',root),target=new URL('ui/test-results/inbox-design/',repo);await mkdir(target,{recursive:true});
const input=await readFile(new URL('current-fixture.tsx',root),'utf8');const revision=createHash('sha256').update(input).update(await readFile(new URL('opendesign.css',root)).catch(()=>Buffer.from(''))).update(await readFile(new URL('layout-fixes.css',root)).catch(()=>Buffer.from(''))).digest('hex').slice(0,12);
await writeFile(new URL('current.tsx',target),input.replaceAll('/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor',repo.pathname.replace(/\/$/,'')).replaceAll('REVISION',revision));
await writeFile(new URL('current.html',target),'<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inbox design preview</title><div id="root"></div><script type="module" src="./current.tsx?rev='+revision+'"></script></html>');
for(const name of ['opendesign.css','layout-fixes.css'])await copyFile(new URL(name,root),new URL(name,target)).catch(()=>writeFile(new URL(name,target),''));
console.log('Prepared real React Inbox fixture');

await copyFile(new URL('index.html',root),new URL('index.html',target));
