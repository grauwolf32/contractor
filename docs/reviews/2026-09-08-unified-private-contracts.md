# Единый private contract v1alpha1 — V50-001

Проверено 2026-09-08. Текущие контракты Server ↔ Runtime перенесены под исходные
имена без сохранения прежних моделей и преобразований между версиями.

## Результат

- Единственный маркер документа — `apiVersion: contractor/v1alpha1`.
  Поле `privateProtocolVersion`, типы с суффиксом `V2` и прежние DTO удалены.
- Схемы находятся в `api/v1alpha1`, общие Go/Python fixtures — в
  `api/testdata/v1alpha1`. Каталоги `api/private-v2` и
  `testdata/contracts/private-v2` удалены. Обновлены команды и ссылки.
- `AgentRegistration` требует метки и список адаптеров, ответ содержит
  authoritative principal/labels/revision. `AllocationSpec` всегда содержит
  `resolvedRuntimeConfigProvenance` и текущие `RuntimeSettings`.
- Подготовка allocation всегда использует текущий lifecycle адаптеров.
  Проверки наличия workspace, настройки HTTP/Caido и сокрытие секретов
  больше не зависят от принадлежности к дополнительному типу `V2`.
- Сохранены строгий JSON, отказ на неизвестных полях и повторяющихся ключах,
  каноническая сериализация, безопасные ошибки и изоляция повреждённых
  необязательных resource metrics. Общие фрагменты схем сохраняют ограничения
  идентификаторов, namespace, ModelPolicy и точных Artifact refs.

Новые отрицательные fixtures проверяют отсутствие обязательных меток,
адаптеров и provenance; отдельная проверка исключает повторяющиеся `$id` схем.
Тестовые настройки используют данные из общего каталога. Полный прогон также
выявил устаревшие ожидания каталога AgentSkills и тестового Podman workflow:
список шаблонов и разрешение ModelPolicy приведены к текущим конфигурациям.

## Проверка

Итоговый снимок задачи проверен в отдельном checkout поверх актуального main,
включая закоммиченный V51-001. Незавершённые правки документации из общего
рабочего каталога не входят в снимок.

- `go test ./...` — успешно; PostgreSQL-зависимые тесты без заданной тестовой БД
  используют штатный skip.
- `cd runtime && uv run --frozen pytest --tb=short` — 1775 passed, 30 skipped.
- `go test -tags=integration -count=1 -timeout=2m ./internal/controlplane -run
  '^TestCrossLanguageMTLSAllocationLifecycle$'` — реальные Go/Python процессы
  прошли prepare, finalize/abort и release по mTLS.
- `make verify-wire-contracts verify-public-api` — успешно, включая 171 Python-тест.
- `tests/test_audit_completion_contracts.py` — успешно после объединения
  общих schema fragments.
- `tests/test_otlp_retries.py` — 50 passed; проверяется также текущая схема
  полного final response с метриками адаптеров.
- `make lint` и `git diff --check` — успешно.

Миграции сохранённых данных и проверки обновления развёрнутых сервисов не входят
в эту задачу: пользователь подтвердил, что код находится в локальной разработке.
