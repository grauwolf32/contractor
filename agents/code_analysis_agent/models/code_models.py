from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel, Field

# =========================
# Общие типы / примитивы
# =========================


class ArtifactRef(BaseModel):
    """Ссылка на файл/фрагмент в репозитории."""

    path: str
    lines: Dict[str, int] = Field(default_factory=dict)  # {'start': int, 'end': int}
    snippet: Optional[str] = ""
    comment: str = ""


class UrlRef(BaseModel):
    url: AnyUrl


ArtifactOrUrl = Union[ArtifactRef, UrlRef]


class RepoNode(BaseModel):
    path: str
    kind: Literal["file", "dir"]
    comment: str = ""
    children: List["RepoNode"] = Field(default_factory=list)


class ModuleRef(BaseModel):
    name: str
    path: str
    responsibilities: List[str] = Field(default_factory=list)
    deps: List[str] = Field(default_factory=list)


class SchemaRef(BaseModel):
    name: str
    path: str
    kind: str = ""  # e.g., "pydantic", "sqlalchemy", "proto", ...


class CryptoItem(BaseModel):
    name: str
    path: str
    operation: Literal[
        "encrypt", "decrypt", "sign", "verify", "jwt-encode", "jwt-decode", "other"
    ]
    details: str = ""


class IntegrationRef(BaseModel):
    name: str
    path: str
    protocol: str = ""  # e.g., "https", "amqp"
    sdk: str = ""  # e.g., "boto3"
    notes: str = ""


class DbRef(BaseModel):
    technology: str  # e.g., "SQLite", "PostgreSQL", "MongoDB"
    paths: List[ArtifactRef] = Field(default_factory=list)
    migrations: List[ArtifactRef] = Field(default_factory=list)
    orm: str = ""  # e.g., "SQLAlchemy", "Django ORM"


class AuthInfo(BaseModel):
    required: bool
    scheme: str  # e.g., "jwt", "basic", "api-key"


class EndpointRef(BaseModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    path: str
    handler: ArtifactRef
    request_model: str = ""
    response_model: str = ""
    auth: AuthInfo | None = None
    notes: str = ""


class RpcRef(BaseModel):
    service: str
    path: str
    framework: Literal["grpc", "thrift", "other"] = "other"
    proto_or_idl: ArtifactRef | None = None
    handlers: List[ArtifactRef] = Field(default_factory=list)


class BrokerRef(BaseModel):
    technology: str  # Kafka, RabbitMQ, SQS, NATS, ...
    producers: List[ArtifactRef] = Field(default_factory=list)
    consumers: List[ArtifactRef] = Field(default_factory=list)
    topics_or_queues: List[str] = Field(default_factory=list)


class GraphQLRef(BaseModel):
    schema: ArtifactOrUrl  # файл или URL со схемой
    resolvers: List[ArtifactRef] = Field(default_factory=list)


# =========================
# Секционные схемы (независимые)
# =========================


class ConfigurationAndSettings(BaseModel):
    """1. Конфигурация и настройки (.env, config.yaml, properties, и пр.)."""

    locations: List[ArtifactRef] = Field(default_factory=list)
    env_files: List[ArtifactRef] = Field(default_factory=list)
    notes: str = ""


class TestsSection(BaseModel):
    """2. Тесты: расположение, виды, фреймворки."""

    locations: List[ArtifactRef] = Field(default_factory=list)
    kinds_present: Dict[str, bool] = Field(
        default_factory=lambda: {"unit": False, "integration": False, "e2e": False}
    )
    frameworks: List[str] = Field(default_factory=list)


class BusinessLogicSection(BaseModel):
    """3. Бизнес-логика: ключевые модули/слои."""

    modules: List[ModuleRef] = Field(default_factory=list)
    notes: str = ""


class SerializationItem(SchemaRef):
    """Элементы для сериализационных схем (Proto/XSD/Avro/Thrift/JSON Schema)."""

    format: Literal["protobuf", "xsd", "avro", "thrift", "jsonschema", "other"]


class DataModelsAndSchemas(BaseModel):
    """4. Модели данных и схемы."""

    model_definitions: List[SchemaRef] = Field(
        default_factory=list
    )  # ORM/Pydantic/etc.
    serialization_schemas: List[SerializationItem] = Field(default_factory=list)


class CryptographySection(BaseModel):
    """5. Криптография: шифрование/подпись/JWT."""

    encryption_decryption: List[CryptoItem] = Field(default_factory=list)
    signatures: List[CryptoItem] = Field(default_factory=list)
    jwt: List[CryptoItem] = Field(default_factory=list)


class AuthBlock(BaseModel):
    mechanisms: List[str] = Field(default_factory=list)
    files: List[ArtifactRef] = Field(default_factory=list)


class AuthorizationBlock(BaseModel):
    models: List[str] = Field(default_factory=list)
    policies: List[str] = Field(default_factory=list)
    files: List[ArtifactRef] = Field(default_factory=list)


class SecuritySection(BaseModel):
    """6. Безопасность: аутентификация/авторизация/политики/мидлвары."""

    authentication: AuthBlock
    authorization: AuthorizationBlock
    access_control_enforcement: List[ArtifactRef] = Field(default_factory=list)
    security_middleware: List[ArtifactRef] = Field(default_factory=list)


class ExternalIntegrations(BaseModel):
    """7. Интеграции: внешние клиенты, БД, HTTP, RPC, брокеры, WS, GraphQL, SOAP."""

    external_clients: List[IntegrationRef] = Field(default_factory=list)
    databases: List[DbRef] = Field(default_factory=list)
    http_handling: List[EndpointRef] = Field(default_factory=list)
    grpc_thrift: List[RpcRef] = Field(default_factory=list)
    message_brokers: List[BrokerRef] = Field(default_factory=list)
    websockets: List[ArtifactRef] = Field(default_factory=list)
    graphql: List[GraphQLRef] = Field(default_factory=list)
    soap: List[ArtifactRef] = Field(default_factory=list)


class DocumentationSection(BaseModel):
    """8. Документация: README, /docs, wiki и т.п."""

    locations: List[ArtifactRef] = Field(default_factory=list)


class ApiSpecificationSection(BaseModel):
    """9. API спецификации: OpenAPI/Swagger, GraphQL."""

    openapi_schema: List[ArtifactOrUrl] = Field(default_factory=list)
    graphql_schema: List[ArtifactOrUrl] = Field(default_factory=list)
