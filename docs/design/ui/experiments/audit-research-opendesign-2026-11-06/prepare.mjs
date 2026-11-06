import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),repo=new URL('../../../../../',root),target=new URL('ui/test-results/audit-research/',repo);
await mkdir(target,{recursive:true});
const names=['native/index.html','native/audit.css','preview.js','host-polish.css'];
const parts=await Promise.all(names.map(n=>readFile(new URL(n,root),'utf8')));
const revision=createHash('sha256').update(parts.join('\n')).digest('hex').slice(0,12);
const original=parts[0];
// Keep native output immutable. Add only explicit host polish and synthetic wiring.
const html=original.replace('</head>',`<link rel="stylesheet" href="./host-polish.css?rev=${revision}"></head>`).replace('src="./preview.js"',`src="./preview.js?rev=${revision}"`).replace('href="./audit.css"',`href="./audit.css?rev=${revision}"`);
await writeFile(new URL('native.html',target),html);
for(const [from,to] of [['native/audit.css','audit.css'],['native-inputs/theme.css','theme.css'],['host-polish.css','host-polish.css'],['preview.js','preview.js'],['review.js','review.js'],['index.html','index.html']])await copyFile(new URL(from,root),new URL(to,target));
console.log(JSON.stringify({prepared:true,revision,target:target.pathname}));
