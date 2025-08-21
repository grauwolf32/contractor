from typing import Final

code_analysis_schema: Final[str] = """
IMPORTANT: Return your response as a JSON object matching this structure:
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/code-project-analysis.schema.json",
  "title": "CodeProjectAnalysis",
  "type": "object",
  "required": ["meta", "target", "overview", "categories"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["version", "generated_at"],
      "properties": {
        "version": { "type": "string" },
        "generated_at": { "type": "string", "format": "date-time" },
        "tool": { "type": "string" }
      },
      "additionalProperties": false
    },
    "target": {
      "type": "object",
      "required": ["root_path"],
      "properties": {
        "root_path": { "type": "string" },
        "vcs": {
          "type": "object",
          "properties": {
            "type": { "type": "string" },
            "url": { "type": "string" },
            "branch": { "type": "string" },
            "commit": { "type": "string" }
          },
          "additionalProperties": false
        },
        "languages": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "files": { "type": "integer" },
              "sloc": { "type": "integer" }
            },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    },
    "overview": {
      "type": "object",
      "required": ["name", "repo_map"],
      "properties": {
        "name": { "type": "string" },
        "summary": { "type": "string" },
        "repo_map": {
          "type": "array",
          "items": { "$ref": "#/$defs/RepoNode" }
        }
      },
      "additionalProperties": false
    },
    "categories": {
      "type": "object",
      "required": [
        "configuration_and_settings",
        "tests",
        "business_logic",
        "data_models_and_schemas",
        "cryptography",
        "security",
        "external_integrations",
        "documentation",
        "api_specification"
      ],
      "properties": {
        "configuration_and_settings": {
          "type": "object",
          "properties": {
            "locations": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
            "env_files": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
            "notes": { "type": "string" }
          },
          "additionalProperties": false
        },
        "tests": {
          "type": "object",
          "properties": {
            "locations": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
            "kinds_present": {
              "type": "object",
              "properties": {
                "unit": { "type": "boolean" },
                "integration": { "type": "boolean" },
                "e2e": { "type": "boolean" }
              },
              "additionalProperties": false
            },
            "frameworks": { "type": "array", "items": { "type": "string" } }
          },
          "additionalProperties": false
        },
        "business_logic": {
          "type": "object",
          "properties": {
            "modules": {
              "type": "array",
              "items": { "$ref": "#/$defs/ModuleRef" }
            },
            "notes": { "type": "string" }
          },
          "additionalProperties": false
        },
        "data_models_and_schemas": {
          "type": "object",
          "properties": {
            "model_definitions": { "type": "array", "items": { "$ref": "#/$defs/SchemaRef" } },
            "serialization_schemas": {
              "type": "array",
              "items": {
                "allOf": [
                  { "$ref": "#/$defs/SchemaRef" },
                  {
                    "type": "object",
                    "properties": {
                      "format": {
                        "type": "string",
                        "enum": ["protobuf", "xsd", "avro", "thrift", "jsonschema", "other"]
                      }
                    },
                    "required": ["format"],
                    "additionalProperties": false
                  }
                ]
              }
            }
          },
          "additionalProperties": false
        },
        "cryptography": {
          "type": "object",
          "properties": {
            "encryption_decryption": { "type": "array", "items": { "$ref": "#/$defs/CryptoItem" } },
            "signatures": { "type": "array", "items": { "$ref": "#/$defs/CryptoItem" } },
            "jwt": { "type": "array", "items": { "$ref": "#/$defs/CryptoItem" } }
          },
          "additionalProperties": false
        },
        "security": {
          "type": "object",
          "properties": {
            "authentication": {
              "type": "object",
              "properties": {
                "mechanisms": { "type": "array", "items": { "type": "string" } },
                "files": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } }
              },
              "additionalProperties": false
            },
            "authorization": {
              "type": "object",
              "properties": {
                "models": { "type": "array", "items": { "type": "string" } },
                "policies": { "type": "array", "items": { "type": "string" } },
                "files": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } }
              },
              "additionalProperties": false
            },
            "access_control_enforcement": {
              "type": "array",
              "items": { "$ref": "#/$defs/ArtifactRef" }
            },
            "security_middleware": {
              "type": "array",
              "items": { "$ref": "#/$defs/ArtifactRef" }
            }
          },
          "additionalProperties": false
        },
        "external_integrations": {
          "type": "object",
          "properties": {
            "external_clients": { "type": "array", "items": { "$ref": "#/$defs/IntegrationRef" } },
            "databases": { "type": "array", "items": { "$ref": "#/$defs/DbRef" } },
            "http_handling": { "type": "array", "items": { "$ref": "#/$defs/EndpointRef" } },
            "grpc_thrift": { "type": "array", "items": { "$ref": "#/$defs/RpcRef" } },
            "message_brokers": { "type": "array", "items": { "$ref": "#/$defs/BrokerRef" } },
            "websockets": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
            "graphql": { "type": "array", "items": { "$ref": "#/$defs/GraphQLRef" } },
            "soap": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } }
          },
          "additionalProperties": false
        },
        "documentation": {
          "type": "object",
          "properties": {
            "locations": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } }
          },
          "additionalProperties": false
        },
        "api_specification": {
          "type": "object",
          "properties": {
            "openapi_swagger": {
              "type": "array",
              "items": { "$ref": "#/$defs/ArtifactOrUrl" }
            },
            "graphql_schema": {
              "type": "array",
              "items": { "$ref": "#/$defs/ArtifactOrUrl" }
            }
          },
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    }
  },
  "$defs": {
    "ArtifactRef": {
      "type": "object",
      "required": ["path"],
      "properties": {
        "path": { "type": "string" },
        "lines": {
          "type": "object",
          "properties": {
            "start": { "type": "integer" },
            "end": { "type": "integer" }
          },
          "additionalProperties": false
        },
        "snippet": { "type": "string" },
        "comment": { "type": "string" }
      },
      "additionalProperties": false
    },
    "ArtifactOrUrl": {
      "oneOf": [
        { "$ref": "#/$defs/ArtifactRef" },
        {
          "type": "object",
          "required": ["url"],
          "properties": { "url": { "type": "string" } },
          "additionalProperties": false
        }
      ]
    },
    "RepoNode": {
      "type": "object",
      "required": ["path", "kind"],
      "properties": {
        "path": { "type": "string" },
        "kind": { "type": "string", "enum": ["file", "dir"] },
        "comment": { "type": "string" },
        "children": { "type": "array", "items": { "$ref": "#/$defs/RepoNode" } }
      },
      "additionalProperties": false
    },
    "ModuleRef": {
      "type": "object",
      "required": ["name", "path"],
      "properties": {
        "name": { "type": "string" },
        "path": { "type": "string" },
        "responsibilities": { "type": "array", "items": { "type": "string" } },
        "deps": { "type": "array", "items": { "type": "string" } }
      },
      "additionalProperties": false
    },
    "SchemaRef": {
      "type": "object",
      "required": ["name", "path"],
      "properties": {
        "name": { "type": "string" },
        "path": { "type": "string" },
        "kind": { "type": "string" }
      },
      "additionalProperties": false
    },
    "CryptoItem": {
      "type": "object",
      "required": ["name", "path"],
      "properties": {
        "name": { "type": "string" },
        "path": { "type": "string" },
        "operation": {
          "type": "string",
          "enum": ["encrypt", "decrypt", "sign", "verify", "jwt-encode", "jwt-decode", "other"]
        },
        "details": { "type": "string" }
      },
      "additionalProperties": false
    },
    "IntegrationRef": {
      "type": "object",
      "required": ["name", "path"],
      "properties": {
        "name": { "type": "string" },
        "path": { "type": "string" },
        "protocol": { "type": "string" },
        "sdk": { "type": "string" },
        "notes": { "type": "string" }
      },
      "additionalProperties": false
    },
    "DbRef": {
      "type": "object",
      "required": ["technology"],
      "properties": {
        "technology": { "type": "string" },
        "paths": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
        "migrations": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
        "orm": { "type": "string" }
      },
      "additionalProperties": false
    },
    "EndpointRef": {
      "type": "object",
      "required": ["path", "handler"],
      "properties": {
        "method": {
          "type": "string",
          "enum": ["GET","POST","PUT","PATCH","DELETE","OPTIONS","HEAD"]
        },
        "path": { "type": "string" },
        "handler": { "$ref": "#/$defs/ArtifactRef" },
        "request_model": { "type": "string" },
        "response_model": { "type": "string" },
        "auth": {
          "type": "object",
          "properties": {
            "required": { "type": "boolean" },
            "scheme": { "type": "string" }
          },
          "additionalProperties": false
        },
        "notes": { "type": "string" }
      },
      "additionalProperties": false
    },
    "RpcRef": {
      "type": "object",
      "required": ["service", "path"],
      "properties": {
        "service": { "type": "string" },
        "path": { "type": "string" },
        "framework": { "type": "string", "enum": ["grpc", "thrift", "other"] },
        "proto_or_idl": { "$ref": "#/$defs/ArtifactRef" },
        "handlers": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } }
      },
      "additionalProperties": false
    },
    "BrokerRef": {
      "type": "object",
      "required": ["technology"],
      "properties": {
        "technology": { "type": "string" },
        "producers": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
        "consumers": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } },
        "topics_or_queues": { "type": "array", "items": { "type": "string" } }
      },
      "additionalProperties": false
    },
    "GraphQLRef": {
      "type": "object",
      "properties": {
        "schema": { "$ref": "#/$defs/ArtifactOrUrl" },
        "resolvers": { "type": "array", "items": { "$ref": "#/$defs/ArtifactRef" } }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```
"""

basic_project_structure: Final[str] = """
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

