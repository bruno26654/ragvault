# ADR 0013 — Python API design

## Decisão
`ragvault.open()` → `KnowledgeBase` com divulgação progressiva (simples → preset/embedding → config completa). Bindings PyO3 abi3-py39; GIL liberado em busca/ingestão/flush/compact (testado com thread ticker); erros Rust viram exceções específicas; panics nunca cruzam FFI silenciosamente. Dados cruzam a fronteira como JSON + ndarray float32 (vetores nunca via JSON). Async via `asyncio.to_thread` sobre chamadas que liberam o GIL — documentado como thread-offload, não reactor nativo.

## Validação
`test_gil_released_during_search`, suíte pytest completa.
