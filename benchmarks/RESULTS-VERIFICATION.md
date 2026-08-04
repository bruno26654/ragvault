# Offline NLI verification — NOT YET MEASURED

`python benchmarks/bench_nli_verifier.py` could not run here: the model could
not be loaded from this environment.

    OSError: Can't load the configuration of 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2m

**No numbers are published for this adapter yet.** The benchmark, its dataset
and the harness are committed and reproducible; what is missing is a machine
that can reach the model. Run it where huggingface.co is reachable:

    pip install "ragvault[nli]"
    python benchmarks/bench_nli_verifier.py --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

Until that has been run, `ragvault.nli` should be treated as unmeasured:
usable in `report` and `annotate` mode, where a wrong verdict is visible and
costs nothing, and **not** in `repair`/`strict`, where a false `contradicted`
deletes a correct sentence. The gate is the false-contradicted rate, which
this benchmark reports as its headline column.
