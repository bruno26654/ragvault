# ADR 0009 — GPU / cuVS integration

## Estado
**Implemented-experimental (não validado em hardware).** O sidecar `ragvault.gpu.CagraDenseSearcher` está implementado seguindo a API Python documentada do cuVS, com testes de plumbing usando um cuVS falso (wiring, pós-filtro via DSL nativa, fallback CPU) e um teste real marcado `-m gpu` pronto para runner CUDA. Nenhum benchmark GPU foi executado — números só serão publicados após execução real (runbook em docs/GPU.md).

## Plano registrado
- Extra `ragvault[gpu-cu12]` isolado do pacote CPU.
- Backend trait já neutro (o engine escolhe backend por request), então um `CagraGpu`/`FlatGpu` entra sem mudar a API Python.
- Zero-copy só será declarado com DLPack/CUDA Array Interface validados; traces reportarão cópias.

## Critério de aceitação futuro
Benchmarks reais em runner GPU, custo de transferência CPU↔GPU reportado separadamente.
