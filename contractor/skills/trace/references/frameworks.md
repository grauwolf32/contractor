---
description: Framework-specific patterns for tracing request flows — Django, Spring Boot, Go/gorilla-mux. How to follow controller → service → repository call chains efficiently.
---

# Framework-Specific Tracing Patterns

When tracing across multi-file architectures, use these patterns
to follow the call chain without excessive file reads.

## Spring Boot (Java)

### Routing
- `@RestController` + `@RequestMapping("/prefix")` at class level
- `@GetMapping`, `@PostMapping`, `@PutMapping`, `@DeleteMapping` on methods
- Full path = class prefix + method path

### Controller → Service → Repository chain
Spring uses `@Autowired` dependency injection. When you see:
```java
@Autowired UserService userService;
```
The implementation is typically at `<ServiceName>Impl` in a
`service/Impl/` directory. For example:
- `UserService` → `service/Impl/UserServiceImpl.java`
- `OtpService` → `service/Impl/OtpServiceImpl.java`
- `ProfileService` → `service/Impl/ProfileServiceImpl.java`

**After reading the controller method, ALWAYS follow the service
call into the Impl class.** Most vulnerabilities in Spring apps
live in the service layer, not the controller.

### Security config
- `WebSecurityConfig.java` defines URL-level auth rules:
  - `.permitAll()` = no auth required
  - `.authenticated()` = any valid JWT
  - `.hasRole("ADMIN")` = role check
- Check if the target endpoint's path matches a `permitAll()` rule

### JWT validation
- `JwtProvider.java` or similar handles token parsing
- Check for `alg:none` fallback, expired token acceptance,
  signature-less parsing (`PlainJWT.parse`)

## Django / Django REST Framework (Python)

### Routing
- `urls.py` maps URL patterns to view classes
- View classes have `get()`, `post()`, `put()`, `delete()` methods
- `@jwt_auth_required` decorator gates authentication

### Missing auth detection
When a view method LACKS `@jwt_auth_required` but sibling methods
on the same class have it, that's a missing-auth vulnerability.
Compare decorators across methods within the same class.

### Service calls
Django views often contain business logic inline (no separate
service layer). The vulnerable code is usually in the view method
itself. Look for:
- `connection.cursor()` + raw SQL concatenation → SQLi
- `requests.get(user_input)` → SSRF
- `os.path.join(base, user_input)` → path traversal
- `user_details.field += request.data["field"]` → mass assignment

### Large views.py files
Workshop-style Django apps put all views in one large file.
Use `list_symbols` to index the file first, then read only the
relevant method's line range.

## Go / gorilla-mux

### Routing
- `router.HandleFunc("/path", middlewares.SetMiddlewareAuthentication(controller.Handler, db))`
- Handler functions are methods on a `Server` struct:
  `func (s *Server) HandlerName(w http.ResponseWriter, r *http.Request)`

### Middleware
- `SetMiddlewareAuthentication` wraps handlers requiring JWT
- `SetMiddlewareJSON` sets content type
- If a handler is NOT wrapped in `SetMiddlewareAuthentication`,
  it has no auth

### Data flow
Go handlers typically:
1. Read `r.Body` with `io.ReadAll`
2. Unmarshal into a struct or `bson.M`
3. Call model functions that query MongoDB
- Watch for raw `bson.M` from user input passed to
  `collection.FindOne(ctx, bsonMap)` → NoSQL injection

### Annotation comment marker
Go uses `//` comments, not `#`:
```go
// @trace target=HandlerName args=body:tainted calls=ValidateCode
func (s *Server) HandlerName(w http.ResponseWriter, r *http.Request) {
```

## Express / NestJS (Node)

### Routing
- Express: `app.get`/`post`/`put`/`delete("/path", handler)`,
  `router.use("/prefix", subRouter)` for mounting
- NestJS: `@Controller("/prefix")` + `@Get()`/`@Post()`/… on methods;
  full path = controller prefix + method path

### Sources
- `req.query`, `req.body`, `req.params`, `req.headers`, `req.cookies`
  are all tainted (NestJS: `@Query`/`@Body`/`@Param`/`@Headers` params)

### Auth detection
Auth lives in `app.use(mw)` / `router.use(mw)` / per-route middleware
arguments, or NestJS `@UseGuards(...)`. **Open the middleware/guard —
the name (`auth`, `requireUser`) is not evidence.** A route whose
handler array omits the auth middleware that siblings include is a
missing-auth candidate.

### Sinks
- `res.redirect(userInput)` → http.redirect (open redirect)
- `child_process.exec`/`execSync(str)` → shell.exec
- `sequelize.query(str)` raw → db.query.raw; ORM finders = db.query
- `obj[key]=v` / `_.merge` with tainted key → reflect.proto

## Laravel (PHP)

### Routing
- `routes/web.php` / `routes/api.php`: `Route::get`/`post(...)`
- Auth/scoping via `->middleware("auth")` and `Route::group([...], fn)`
  — open the middleware group covering the route before marking auth
  `absent`

### Sinks
- `DB::raw($x)` / `->whereRaw($x)` / `DB::select("... $x ...")` →
  db.query.raw (SQLi); query builder methods (`->where("col", $x)`) =
  db.query (bound)
- `Model::create($request->all())` / `->fill($request->all())` /
  `->update($request->all())` → mass assignment (db.orm.bulk) unless the
  model has a `$fillable`/`$guarded` allowlist
- `view($name)` with tainted `$name` → template selection;
  `{!! $x !!}` (unescaped Blade) → template.render.raw / XSS

## Multi-Service Architectures

When the target endpoint lives behind a reverse proxy (nginx)
routing to multiple services:

1. **Identify which service handles the route** from the URL prefix
   (e.g., `/identity/` → identity service, `/workshop/` → workshop)
2. **Auth typically lives in a different service** than the business
   logic — trace the auth check separately from the data flow
3. **Cross-service calls** (e.g., workshop calling identity's
   `/verify` endpoint) are boundaries — annotate up to the boundary,
   note the cross-service call, but don't trace into the other service

## Tracing Priority

For each handler, trace in this order:
1. **Entrypoint** — the handler method itself (ALWAYS annotate)
2. **Auth check** — decorator, middleware, or inline check
3. **Input parsing** — how request data enters the handler
4. **Service method** — the `@Autowired` service call (ALWAYS follow)
5. **Sink** — where tainted data reaches a sensitive operation
6. **Response** — what data is returned (check for PII exposure)
