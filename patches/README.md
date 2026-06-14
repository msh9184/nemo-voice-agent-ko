# NeMo core patch — cache-aware streaming ASR

`nemo-streaming-asr.patch` contains the changes to **NVIDIA NeMo** core needed to run the
Conformer/RNNT encoder as a correct cache-aware **streaming** encoder for this voice agent,
plus multilingual language-token support.

## What it changes

| File | Purpose |
|------|---------|
| `nemo/collections/asr/modules/conformer_encoder.py` | Chunked forward with `cache_last_channel/time/len`; trace-friendly mask/chunk helpers |
| `nemo/collections/asr/parts/submodules/multi_head_attention.py` | Reconcile query/key-length mismatches introduced by cache slicing during streaming |
| `nemo/collections/asr/parts/submodules/conformer_modules.py` | Fix `pad_mask` dimensions for chunked streaming |
| `nemo/collections/asr/parts/submodules/causal_convs.py` | Guard conv-cache/input shape mismatches |
| `nemo/collections/asr/modules/rnnt.py` | Language-token support for multilingual models |

## Applying

```bash
cd <your-NeMo-checkout>
git apply /path/to/nemo-voice-agent-ko/patches/nemo-streaming-asr.patch
```

The patch is generated against the NeMo revision this project was developed on; if it does not
apply cleanly to your NeMo version, apply the hunks manually (they are small and localized).
