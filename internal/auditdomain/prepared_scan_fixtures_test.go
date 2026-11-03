package auditdomain

const scanSource = `{
 "openapi":"3.1.0","servers":[{"url":"https://unselected.test"}],
 "paths":{
  "/pets/{id}":{"post":{
   "parameters":[{"in":"path","name":"id","required":true}],
   "security":[{"bearer":[]}],
   "requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object"}}}}
  }},
  "/missing/{id}":{"get":{"parameters":[{"in":"path","name":"id","required":true}]}},
  "/unselected":{"get":{}}
 },
 "components":{"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}}}
}`

const sqlmapSettings = `{
 "schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"https://target.test/api",
 "authentication":{"bearer":"never-copy-credential"},"testParameters":["q"],
 "operations":{"#/paths/~1pets~1{id}/post":{
  "parameters":{"path:id":7},"body":{"mediaType":"application/json","value":{"q":"abc"}}
 }}
}`

const nucleiSettings = `{
 "schema":"contractor.audit.openapi-scan-settings.v1","scanner":"nuclei","server":"https://target.test/api",
 "operations":{
  "#/paths/~1pets~1{id}/post":{"parameters":{"path:id":7}},
  "#/paths/~1missing~1{id}/get":{}
 }
}`
