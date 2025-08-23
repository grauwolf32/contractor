from typing import Final

code_analysis_schema: Final[
    str
] = """
IMPORTANT: Return your response as a JSON object matching this structure:
```json

```
"""

basic_project_structure: Final[
    str
] = """
You need to analyze the structure of the project: what directories exist, what files are located inside them, and what their responsibilities are.

The analysis should identify the following categories:

1. Configuration and settings:
   * Where configuration files and environment settings are located (e.g., `.env`, `config.yaml`, `application.properties`, etc.).

2. Tests:
   * Where test files are located.
   * Whether unit tests, integration tests, and end-to-end (E2E) tests are present.

3. Business logic:
   * Where the core application/business logic is implemented.

4. Data models and schemas:
   * Where data models are defined.
   * Where proto-schemas (Protobuf), XSD schemas (XML), or other serialization formats are defined.

5. Cryptographic functions: (if present in the project):
   * Data encryption and decryption.
   * Signature validation and generation.
   * JSON Web Token (JWT) operations.

6. Security-related functions:
   * Authentication logic.
   * Authorization logic.
   * Role models and access control policies.
   * Access control enforcement.
   * Middleware responsible for enforcing security checks.

7. Integration with external systems:
   * Client code for external services (e.g., REST clients, SDKs).
   * Database interaction logic (repositories, DAOs, migrations).
   * HTTP request handling (controllers, routers, request/response logic).
   * gRPC/Thrift service definitions and handlers.
   * Message broker integrations (Kafka, RabbitMQ, other queue systems).
   * WebSocket handling.
   * GraphQL schema definitions and resolvers.
   * SOAP service integrations.

8. Documentation:
   * Where project documentation is stored (e.g., `README.md`, `/docs` folder, wiki).

9. API specification:
   * Where the OpenAPI/Swagger specification (if present) is located.
   * GraphQL Schema

"""
