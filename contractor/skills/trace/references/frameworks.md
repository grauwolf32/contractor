---
description: Framework-specific patterns for tracing request flows — Spring Boot, Django, Go/gorilla-mux, Express/NestJS, Laravel, WordPress. Identify each framework's routing and authorization-control primitive; follow controller → service → repository chains efficiently.
---

# Framework-Specific Tracing Patterns

When tracing across multi-file architectures, use these patterns
to follow the call chain without excessive file reads.

## Recognizing the authorization control primitive (any framework)

Every framework expresses access control through a **primitive** — a
decorator, middleware, route guard, policy attribute, or an inline
capability call. To evaluate authorization on a handler:

1. **Identify the framework's primitive** (the per-framework sections
   below name it: Spring `.hasRole`/security config, Django
   `@…_required`, Express/Nest middleware/`@UseGuards`, Laravel
   `->middleware`, WordPress `current_user_can`).
2. **Check the sensitive handler actually invokes it on the path that
   reaches the operation** (per the dominance rule in
   `trace/references/controls`).

A privileged or state-changing handler that **never invokes the
primitive** is a missing-authorization finding (**Shape B**,
CWE-862/285/269) — reportable from the code structure ALONE; you do NOT
need a tainted-data flow or a demonstrated exploit. "No `current_user_can`
/ no guard / no `permission_callback` on a sensitive action" IS the bug.

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

## WordPress / WP plugins (PHP)

### Routing — hooks & actions, not a route table
WP plugins register entrypoints via hooks. Grep for these, not for routes:
- **AJAX**: `add_action('wp_ajax_{action}', 'handler')` (logged-in) and
  `add_action('wp_ajax_nopriv_{action}', 'handler')` — the `_nopriv_`
  variant is reachable **UNAUTHENTICATED**. Invoked via
  `POST /wp-admin/admin-ajax.php` with `action={action}`.
- **REST**: `register_rest_route('ns/v1', '/path', ['callback'=>'fn',
  'permission_callback'=>...])`. A `permission_callback` of
  `'__return_true'` (or omitted) = **no authorization**.
- **Forms/init**: `add_action('admin_post_{action}'|'admin_post_nopriv_{action}', …)`,
  `add_action('init'|'admin_init', …)`, `add_shortcode(…)`.

### Control primitives (open these before marking authz/csrf present)
- **authentication**: `is_user_logged_in()`, or the action is only on
  `wp_ajax_` (not `_nopriv_`).
- **authorization**: `current_user_can('capability')`
  (`'manage_options'` for admin actions). A sensitive handler with **no
  `current_user_can`** = missing authorization (Shape B, CWE-862/269).
- **CSRF/nonce**: `check_ajax_referer(…)`, `wp_verify_nonce(…)`,
  `check_admin_referer(…)`. Absent on a state-changing AJAX/admin
  handler = CSRF (Shape B, `csrf`).
- A registration / profile-update handler that sets `role`,
  `wp_update_user`, `add_user_to_blog`, or a `*_capabilities` / `is_admin`
  field from request input **without** a `current_user_can` check →
  **privilege escalation** (Shape B, CWE-269).

### Sinks
- `$wpdb->query("… $x …")` / `$wpdb->get_results($raw)` → `db.query.raw`
  (SQLi). `$wpdb->prepare(…)` / `$wpdb->query($wpdb->prepare(…))` = bound.
- `update_option` / `update_user_meta` / `wp_update_user` from request
  input → state change; verify authz + nonce dominate the call.

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
