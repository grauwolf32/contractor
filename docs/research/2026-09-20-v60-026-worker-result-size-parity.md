# V60-026 — размер типизированных Worker results в Go и Python

Проверено 20.09.2026 в изолированном worktree от
`8edeabf21194f5bba8dd258d53a072af713c630e`. Найденный в V60-006 дефект
подтверждён настоящими Go/Python readers и исправлен без изменения числовых
лимитов. Команды, первоначальные неудачи и итоговые результаты находятся в
[evidence](../../tasks/evidence/v60-026.json).

## Воспроизведение и причина

WorkerCompletion с `result` из 65 536 символов `<`, `>` или `&` занимает
65 769 байт compact UTF-8 JSON. Python `decode_private` принимает документ.
Go `DecodeStrict` отвергал идентичные bytes: `WorkerCompletion.Validate`
повторно сериализовал объект через `json.Marshal`, превращая каждый такой
символ в шесть ASCII bytes. Временная копия занимала 393 449 байт и превышала
256 KiB. После исправления все три исходных документа принимаются обоими readers.

ASCII и кириллица той же UTF-8 длины принимаются до и после изменения.
Реальные управляющие символы требуют JSON escaping: их encoded overflow
по-прежнему отклоняется. Дополнительный shared case доказал аналогичное
расхождение для настоящего U+2028 около полного envelope cap.

Основание — общий wire contract в spec14 и явно сохранённые в V57-004 реальные
text/result/wire bounds. Разница необязательного escaping двух стандартных
serializers не должна менять допустимость одного Worker result.

Совпадение учёта размера проверено для поддерживаемых типизированных
`WorkerCompletion`/`WorkerResult` и `StageContentResult`. Их числовые поля —
целочисленные счётчики; произвольные JSON payloads с float-полями не входят в
этот контракт. Результат не является гарантией одинаковой сериализации или
размера любого Go/Python JSON-объекта.

## Исправление

`contracts.ResultJSONSize` использует обычный `json.Encoder` с
`SetEscapeHTML(false)`. Для подсчёта размера устраняется только дополнительный
JavaScript escaping настоящих U+2028/U+2029. Небольшой проход пропускает
экранированные backslashes, поэтому буквальные `\u2028`/`\u003c` остаются
обычными текстовыми последовательностями. Все цифры uint64 считаются точно;
декодирования чисел через float64 нет.

Общий подсчёт применяется к трём уже существующим result boundaries:
`WorkerCompletion`, `WorkerResult` в Streamline и `StageResult` в Planner.
Последние два иначе продолжали бы отвергать результат после успешного A2A.
Размеры текстов и envelope, ordinary finalizer, summarizer, model budgets,
request bounds, private-wire encoders/digests и transport остаются прежними.

## Проверка

- Общая матрица из 18 типизированных WorkerCompletion выполняется обоими
  языками: HTML, Unicode, настоящие и буквальные escapes, quotes/backslashes, uint64 max, текст
  65 536/65 537 bytes и envelope 262 144/262 145 bytes.
- Реальный loopback HTTP с официальным Go A2A SDK передаёт HTML result из
  63 000 bytes с точным сохранением текста и invocation ID.
- Полный Passthrough Runner и Streamline result gate принимают HTML и
  отвергают настоящий encoded overflow управляющих символов.
- `go test -race -count=1` для contracts, planner, streamline и a2a прошёл:
  568 PASS events, включая родительские tests/subtests, без skips.
- `make verify-wire-contracts test-wire-cross-language` прошёл; Python части
  содержат 193 и 114 успешных случаев, включая новую общую матрицу.

Первый focused post-fix запуск обнаружил опечатку в новом тесте: несуществующий
`AsError` заменён штатным `FailureFrom`. Это отражено в evidence; окончательный
полный race-запуск выполнен уже после исправления. Production дефектов в этой
доработке больше не установлено. Проверка не повышает общий контракт размера
и не утверждает, что любые provider/model limits допускают тот же объём.
