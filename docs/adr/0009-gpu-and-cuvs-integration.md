# ADR 0009 — GPU / cuVS integration

## Estado
**Blocked/planned.** Este ambiente não tem GPU; nenhuma integração foi iniciada para não criar stubs enganosos ou benchmarks inventados.

## Plano registrado
- Extra `ragvault[gpu-cu12]` isolado do pacote CPU.
- Backend trait já neutro (o engine escolhe backend por request), então um `CagraGpu`/`FlatGpu` entra sem mudar a API Python.
- Zero-copy só será declarado com DLPack/CUDA Array Interface validados; traces reportarão cópias.

## Critério de aceitação futuro
Benchmarks reais em runner GPU, custo de transferência CPU↔GPU reportado separadamente.
