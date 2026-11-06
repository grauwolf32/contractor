import {readFile,writeFile,mkdir} from 'node:fs/promises';import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),target=new URL('../../../../../ui/test-results/audit-research-production/',root);await mkdir(target,{recursive:true});
const files=['index.html','review.js','current.html','screen.tsx'],sources=await Promise.all(files.map(n=>readFile(new URL('production/'+n,root),'utf8')));
const revision=createHash('sha256').update(sources.join('')).digest('hex').slice(0,12);
for(let i=0;i<files.length;i++)await writeFile(new URL(files[i],target),sources[i].replaceAll('production-1',revision));
console.log(JSON.stringify({target:target.pathname,revision}));

await writeFile(new URL("shared-config.json",target),await readFile(new URL("../../../../../api/testdata/v1alpha1/valid/shared-config.json",root)));
