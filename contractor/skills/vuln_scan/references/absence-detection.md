---
description: Patterns for detecting security vulnerabilities defined by ABSENCE of controls — missing auth, missing ownership checks, missing rate limiting, missing role checks. These are harder to find than positive patterns.
---

# Absence Detection Patterns

Most real-world vulnerabilities are about what's MISSING, not what's
present. This reference covers how to systematically find missing
security controls.

## Method: Compare Siblings

The fastest way to find missing controls is to compare handlers in
the same file/class. If 3 out of 4 methods have auth but one doesn't,
the one without it is likely a bug.

## Missing Authentication

### Django/DRF
```python
# PATTERN: class with mixed auth decorators
class OrderView(APIView):
    @jwt_auth_required      # ← has auth
    def post(self, request, user):
        ...
    def get(self, request):  # ← NO AUTH — vulnerability!
        ...
```
**Detection**: `list_symbols` on views.py → for each class, read ALL
method decorators. Flag methods without @jwt_auth_required when
siblings have it.

### Spring Boot
```java
// In WebSecurityConfig:
.requestMatchers("/api/admin/**").hasRole("ADMIN")
.requestMatchers("/api/user/**").authenticated()
.requestMatchers("/api/public/**").permitAll()
// CHECK: is /api/v2/user/dashboard in permitAll()? → vuln
```
**Detection**: Read WebSecurityConfig.java, map each .requestMatchers
pattern to its rule. Flag permitAll() on sensitive paths.

### Go / gorilla-mux
```go
// With auth:
router.HandleFunc("/api/posts", middlewares.SetMiddlewareAuthentication(
    controller.GetPost, server.DB)).Methods("GET")
// Without auth — vulnerability!
router.HandleFunc("/community/home", middlewares.SetMiddlewareJSON(
    controller.Home)).Methods("GET")
```
**Detection**: grep for `HandleFunc` → check if wrapped in
`SetMiddlewareAuthentication`. No wrapper = no auth.

## Missing Ownership Check (BOLA/IDOR)

Pattern: handler fetches resource by ID from URL path but never
checks that the authenticated user owns it.

### Django
```python
# VULNERABLE — no ownership check:
def get(self, request, order_id):
    order = Order.objects.get(id=order_id)  # any order!
    return Response(OrderSerializer(order).data)

# SAFE — ownership verified:
def get(self, request, order_id):
    order = Order.objects.get(id=order_id, user=request.user)
```
**Detection**: Find handlers with path params ({id}, {order_id}).
Read the DB query — does it include user/owner filter?

### Spring Boot
```java
// VULNERABLE:
@GetMapping("/vehicle/{id}/location")
public VehicleLocation get(@PathVariable Long id) {
    return vehicleRepo.findById(id);  // no owner check!
}

// SAFE:
return vehicleRepo.findByIdAndOwnerId(id, currentUser.getId());
```

### Go
```go
// VULNERABLE:
func (s *Server) GetPostByID(w http.ResponseWriter, r *http.Request) {
    id := mux.Vars(r)["postID"]
    post, _ := models.GetPostByID(s.Client, id)  // no owner check
}
```

## Missing Role Check (BFLA)

Pattern: admin/management endpoint that requires auth but not
admin role.

```python
# VULNERABLE — any authenticated user can access:
class AdminUserView(APIView):
    @jwt_auth_required
    def get(self, request, user):
        return Response(UserDetails.objects.all())  # lists ALL users!
        
# SAFE — role check:
    @jwt_auth_required
    def get(self, request, user):
        if user.role != 'ADMIN':
            return Response(status=403)
```
**Detection**: Find endpoints under /admin/, /management/, or with
"Admin" in class name. Check for role verification after auth.

## Missing Rate Limiting

Pattern: auth-sensitive endpoints (login, OTP, password reset)
without throttling.

**Detection**:
1. Search project for rate-limit libraries:
   - Python: `slowapi`, `flask-limiter`, `django-ratelimit`, `throttle`
   - Java: `@RateLimiter`, bucket4j, resilience4j
   - Go: `rate.Limiter`, `tollbooth`
   - Node: `express-rate-limit`, `rate-limiter-flexible`
2. If none found → report for ALL auth endpoints
3. If found → check if applied to login/OTP/reset specifically

## Missing Input Validation (Mass Assignment)

Pattern: request body passed directly to model create/update.

```python
# VULNERABLE — user-supplied amount:
user.available_credit += request.data["amount"]  # attacker controls amount!

# VULNERABLE — full body to model:
User.objects.create(**request.data)  # attacker adds is_admin=True

# SAFE — explicit fields only:
User.objects.create(name=data["name"], email=data["email"])
```
**Detection**: Find create/update/save calls. Check if request
data dict is spread into them or if individual fields are cherry-picked.

## User Enumeration

Pattern: different error responses for valid vs invalid users.

```python
# VULNERABLE — different messages:
if user is None:
    return Response({"error": "User not found"}, status=404)
else:
    if not check_password(user, password):
        return Response({"error": "Invalid password"}, status=401)

# SAFE — same message regardless:
return Response({"error": "Invalid credentials"}, status=401)
```
**Detection**: Read login/forgot-password handlers. Check if error
messages or status codes differ based on user existence.

## Path Traversal (Absence of Path Confinement)

Pattern: file path constructed from user input without checking
it stays within allowed directory.

```python
# VULNERABLE — no confinement check:
path = os.path.join(BASE_DIR, 'reports', user_filename)
return FileResponse(open(path, 'rb'))

# SAFE — realpath + prefix check:
full = os.path.realpath(os.path.join(BASE_DIR, 'reports', user_filename))
if not full.startswith(os.path.realpath(BASE_DIR + '/reports')):
    return Response(status=403)
```
**Detection**: Find file-serving endpoints (download, export, report).
Check if path confinement (realpath + startswith) is applied.

## Systematic Workflow

For each handler file:
1. `list_symbols` → get ALL functions/methods
2. `read_file` first 5-10 lines of each method → check decorators
3. Compare: if any method lacks auth that siblings have → report
4. For data-access methods: check ownership filter → report BOLA
5. For admin-looking methods: check role verification → report BFLA
6. For file-serving methods: check path confinement → report traversal
