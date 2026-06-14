#!/usr/bin/env python3
"""
Utility script to fix .nemo model files for compatibility.

This script modifies model_config.yaml inside .nemo files to:
1. Fix target field to use concrete model class (not abstract ASRModel)
2. Remove incompatible parameters for older NeMo versions
3. Add missing parameters for Hybrid RNNT+CTC models (e.g., aux_ctc.decoder.num_classes)
4. [NEW] Strip CTC components to convert Hybrid model to pure RNNT (--strip-ctc)

Note: sdpa_gate is now supported (implemented in multi_head_attention.py) and preserved in config

Usage:
    python fix_nemo_model.py input.nemo [output.nemo]
    python fix_nemo_model.py input.nemo --strip-ctc  # Convert Hybrid to pure RNNT

    If output.nemo is not specified, saves as input_fixed.nemo

Errors this fixes:
    TypeError: Can't instantiate abstract class ASRModel with abstract methods
    setup_training_data, setup_validation_data

    Missing key num_classes, full_key: decoder.num_classes (for aux_ctc.decoder)

    Note: sdpa_gate is now supported in NeMo source code (no longer an error)

Features:
    --strip-ctc: Remove CTC decoder and aux_ctc config, convert to pure RNNT model
                 This reduces model size and memory footprint for RNNT-only inference
"""

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Any

import yaml

# Optional: torch for checkpoint manipulation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Model architecture to concrete class mapping
MODEL_TARGET_MAP = {
    # FastConformer Hybrid (RNNT + CTC) BPE
    "fastconformer_hybrid_bpe": "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel",
    # FastConformer Hybrid (RNNT + CTC) Char
    "fastconformer_hybrid_char": "nemo.collections.asr.models.hybrid_rnnt_ctc_models.EncDecHybridRNNTCTCModel",
    # FastConformer CTC BPE
    "fastconformer_ctc_bpe": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
    # FastConformer RNNT BPE
    "fastconformer_rnnt_bpe": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    # Conformer Hybrid (RNNT + CTC) BPE - Added for compatibility
    "conformer_hybrid_bpe": "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel",
    # Conformer Hybrid (RNNT + CTC) Char
    "conformer_hybrid_char": "nemo.collections.asr.models.hybrid_rnnt_ctc_models.EncDecHybridRNNTCTCModel",
    # Conformer CTC BPE
    "conformer_ctc_bpe": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
    # Conformer RNNT BPE
    "conformer_rnnt_bpe": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    # Conformer CTC Char
    "conformer_ctc_char": "nemo.collections.asr.models.ctc_models.EncDecCTCModel",
    # Conformer RNNT Char
    "conformer_rnnt_char": "nemo.collections.asr.models.rnnt_models.EncDecRNNTModel",
}

# Parameters to remove for compatibility with older NeMo versions
# These parameters exist in newer NeMo but cause errors in older versions
# Note: sdpa_gate is now supported (implemented in multi_head_attention.py)
INCOMPATIBLE_ENCODER_PARAMS = [
    # "sdpa_gate",         # SDPA gating - NOW SUPPORTED (removed from incompatible list)
    "use_sdpa",            # Use SDPA flag
    "sdpa_kernel_size",    # SDPA kernel size
]

# Decoder-specific incompatible parameters
INCOMPATIBLE_DECODER_PARAMS = [
    "use_language_tokenizer",  # Language tokenizer feature - newer NeMo
    "language_tokenizer",      # Related to language tokenizer
    "num_extra_outputs",       # Extra outputs feature - newer NeMo
]

# Joint module incompatible parameters (RNNT)
INCOMPATIBLE_JOINT_PARAMS = [
    "num_extra_outputs",       # Extra outputs feature - newer NeMo
]

INCOMPATIBLE_MODEL_PARAMS = [
    # Add any model-level incompatible params here if needed
]


def remove_incompatible_params(config: dict) -> dict:
    """
    Remove incompatible parameters from model config for older NeMo versions.

    Args:
        config: Model configuration dictionary

    Returns:
        Modified configuration with incompatible params removed
    """
    removed_params = []

    # Remove from encoder config
    encoder_cfg = config.get("encoder", {})
    if encoder_cfg and isinstance(encoder_cfg, dict):
        for param in INCOMPATIBLE_ENCODER_PARAMS:
            if param in encoder_cfg:
                del encoder_cfg[param]
                removed_params.append(f"encoder.{param}")
        config["encoder"] = encoder_cfg

    # Remove from model-level config
    for param in INCOMPATIBLE_MODEL_PARAMS:
        if param in config:
            del config[param]
            removed_params.append(param)

    # Check preprocessor
    preprocessor_cfg = config.get("preprocessor", {})
    if preprocessor_cfg and isinstance(preprocessor_cfg, dict):
        for param in INCOMPATIBLE_ENCODER_PARAMS:
            if param in preprocessor_cfg:
                del preprocessor_cfg[param]
                removed_params.append(f"preprocessor.{param}")
        config["preprocessor"] = preprocessor_cfg

    # Check decoder - use DECODER-SPECIFIC incompatible params
    decoder_cfg = config.get("decoder", {})
    if decoder_cfg and isinstance(decoder_cfg, dict):
        for param in INCOMPATIBLE_DECODER_PARAMS:
            if param in decoder_cfg:
                del decoder_cfg[param]
                removed_params.append(f"decoder.{param}")
        config["decoder"] = decoder_cfg

    # Check joint (for RNNT models) - use JOINT-SPECIFIC incompatible params
    joint_cfg = config.get("joint", {})
    if joint_cfg and isinstance(joint_cfg, dict):
        for param in INCOMPATIBLE_JOINT_PARAMS:
            if param in joint_cfg:
                del joint_cfg[param]
                removed_params.append(f"joint.{param}")
        config["joint"] = joint_cfg

    # Check aux_ctc decoder (for Hybrid models)
    aux_ctc = config.get("aux_ctc", {})
    if aux_ctc and isinstance(aux_ctc, dict):
        aux_decoder = aux_ctc.get("decoder", {})
        if aux_decoder and isinstance(aux_decoder, dict):
            for param in INCOMPATIBLE_DECODER_PARAMS:
                if param in aux_decoder:
                    del aux_decoder[param]
                    removed_params.append(f"aux_ctc.decoder.{param}")
            aux_ctc["decoder"] = aux_decoder
            config["aux_ctc"] = aux_ctc

    return config, removed_params


def get_vocab_size_from_config(config: dict) -> int:
    """
    Extract vocabulary size from various config locations.

    Args:
        config: Model configuration dictionary

    Returns:
        Vocabulary size or None if not found
    """
    vocab_size = None

    # 1. Try decoder.vocab_size (RNNT decoder)
    decoder_cfg = config.get("decoder", {})
    if isinstance(decoder_cfg, dict) and "vocab_size" in decoder_cfg:
        vocab_size = decoder_cfg["vocab_size"]

    # 2. Try joint.num_classes (RNNT joint)
    if vocab_size is None:
        joint_cfg = config.get("joint", {})
        if isinstance(joint_cfg, dict) and "num_classes" in joint_cfg:
            vocab_size = joint_cfg["num_classes"]

    # 3. Try labels list length
    if vocab_size is None:
        labels = config.get("labels", [])
        if labels and isinstance(labels, list):
            vocab_size = len(labels)

    # 4. Try tokenizer vocabulary
    if vocab_size is None:
        tokenizer_cfg = config.get("tokenizer", {})
        if isinstance(tokenizer_cfg, dict):
            vocab_size = tokenizer_cfg.get("vocab_size")

    return vocab_size


def diagnose_config_structure(config: dict) -> List[str]:
    """
    Diagnose the config structure and return relevant information.

    Args:
        config: Model configuration dictionary

    Returns:
        List of diagnostic messages
    """
    diagnostics = []

    # Check top-level keys
    top_keys = list(config.keys())
    diagnostics.append(f"Top-level keys: {top_keys}")

    # Check decoder structure
    decoder_cfg = config.get("decoder", {})
    if decoder_cfg:
        decoder_keys = list(decoder_cfg.keys()) if isinstance(decoder_cfg, dict) else str(type(decoder_cfg))
        diagnostics.append(f"decoder keys: {decoder_keys}")
        if isinstance(decoder_cfg, dict):
            if "num_classes" in decoder_cfg:
                diagnostics.append(f"  decoder.num_classes = {decoder_cfg['num_classes']}")
            if "vocab_size" in decoder_cfg:
                diagnostics.append(f"  decoder.vocab_size = {decoder_cfg['vocab_size']}")
            if "_target_" in decoder_cfg:
                diagnostics.append(f"  decoder._target_ = {decoder_cfg['_target_']}")
    else:
        diagnostics.append("decoder: NOT FOUND")

    # Check aux_ctc structure
    aux_ctc = config.get("aux_ctc", {})
    if aux_ctc:
        aux_keys = list(aux_ctc.keys()) if isinstance(aux_ctc, dict) else str(type(aux_ctc))
        diagnostics.append(f"aux_ctc keys: {aux_keys}")
        if isinstance(aux_ctc, dict):
            aux_decoder = aux_ctc.get("decoder", {})
            if aux_decoder:
                aux_decoder_keys = list(aux_decoder.keys()) if isinstance(aux_decoder, dict) else str(type(aux_decoder))
                diagnostics.append(f"  aux_ctc.decoder keys: {aux_decoder_keys}")
                if isinstance(aux_decoder, dict) and "num_classes" in aux_decoder:
                    diagnostics.append(f"    aux_ctc.decoder.num_classes = {aux_decoder['num_classes']}")
    else:
        diagnostics.append("aux_ctc: NOT FOUND")

    # Check joint structure
    joint_cfg = config.get("joint", {})
    if joint_cfg:
        joint_keys = list(joint_cfg.keys()) if isinstance(joint_cfg, dict) else str(type(joint_cfg))
        diagnostics.append(f"joint keys: {joint_keys}")
        if isinstance(joint_cfg, dict) and "num_classes" in joint_cfg:
            diagnostics.append(f"  joint.num_classes = {joint_cfg['num_classes']}")
    else:
        diagnostics.append("joint: NOT FOUND")

    # Check interctc
    if "interctc" in config:
        diagnostics.append(f"interctc: FOUND (keys: {list(config['interctc'].keys()) if isinstance(config['interctc'], dict) else 'N/A'})")

    return diagnostics


def fix_hybrid_model_config(config: dict, verbose: bool = False) -> tuple:
    """
    Fix configuration for Hybrid RNNT+CTC models.

    This function handles:
    1. Add num_classes to aux_ctc.decoder if missing (required for ConvASRDecoder)
    2. Add num_classes to main decoder if missing (some configs require this)
    3. Ensure feat_in is set in aux_ctc.decoder

    Args:
        config: Model configuration dictionary
        verbose: If True, print diagnostic information

    Returns:
        Tuple of (modified config, list of fixes applied)
    """
    fixes_applied = []

    # Run diagnostics if verbose
    if verbose:
        print("  [DEBUG] Config structure diagnosis:")
        for diag in diagnose_config_structure(config):
            print(f"    {diag}")

    # Get vocab_size from various sources
    vocab_size = get_vocab_size_from_config(config)
    if verbose:
        print(f"  [DEBUG] Detected vocab_size: {vocab_size}")

    # Get encoder d_model for feat_in
    encoder_cfg = config.get("encoder", {})
    d_model = encoder_cfg.get("d_model") if isinstance(encoder_cfg, dict) else None

    # ===== Fix 1: Main decoder - handle RNNT vs CTC correctly =====
    # IMPORTANT: Do NOT add num_classes to RNNT decoder!
    # - RNNT decoder uses 'vocab_size', not 'num_classes'
    # - CTC decoder (ConvASRDecoder) uses 'num_classes'
    # - In Hybrid models, main 'decoder' is RNNT, 'aux_ctc.decoder' is CTC
    decoder_cfg = config.get("decoder", {})
    if isinstance(decoder_cfg, dict):
        decoder_target = decoder_cfg.get("_target_", "")

        # Check decoder type
        is_ctc_decoder = any(x in decoder_target.lower() for x in ["ctc", "convasr", "conv_asr"])
        is_rnnt_decoder = any(x in decoder_target.lower() for x in ["rnnt", "rnn_t", "transducer"])

        # If this is an RNNT decoder, REMOVE num_classes if it exists (it's wrong!)
        if is_rnnt_decoder and "num_classes" in decoder_cfg:
            del decoder_cfg["num_classes"]
            config["decoder"] = decoder_cfg
            fixes_applied.append("REMOVED decoder.num_classes (RNNT decoder uses vocab_size, not num_classes)")

        # Only add num_classes if it's explicitly a CTC decoder (NOT RNNT)
        elif is_ctc_decoder and not is_rnnt_decoder:
            if "num_classes" not in decoder_cfg and vocab_size is not None:
                decoder_cfg["num_classes"] = vocab_size
                config["decoder"] = decoder_cfg
                fixes_applied.append(f"decoder.num_classes = {vocab_size} (CTC decoder)")

            # Also ensure feat_in is set for CTC decoders
            if "feat_in" not in decoder_cfg and d_model:
                decoder_cfg["feat_in"] = d_model
                config["decoder"] = decoder_cfg
                fixes_applied.append(f"decoder.feat_in = {d_model} (CTC decoder)")

    # ===== Fix 2: aux_ctc.decoder.num_classes (for Hybrid models) =====
    aux_ctc = config.get("aux_ctc", {})
    if isinstance(aux_ctc, dict) and aux_ctc:
        aux_decoder_cfg = aux_ctc.get("decoder", {})

        # Create decoder config if missing but aux_ctc exists
        if not aux_decoder_cfg:
            aux_decoder_cfg = {}
            if verbose:
                print("  [DEBUG] Creating aux_ctc.decoder section")

        if isinstance(aux_decoder_cfg, dict):
            # Add num_classes if missing
            if "num_classes" not in aux_decoder_cfg:
                if vocab_size is not None:
                    aux_decoder_cfg["num_classes"] = vocab_size
                    fixes_applied.append(f"aux_ctc.decoder.num_classes = {vocab_size}")
                else:
                    # Use -1 as placeholder (model will set from vocabulary)
                    aux_decoder_cfg["num_classes"] = -1
                    fixes_applied.append("aux_ctc.decoder.num_classes = -1 (placeholder)")
            elif aux_decoder_cfg.get("num_classes", 0) <= 0 and vocab_size is not None:
                aux_decoder_cfg["num_classes"] = vocab_size
                fixes_applied.append(f"aux_ctc.decoder.num_classes = {vocab_size} (fixed invalid value)")

            # Add feat_in if missing
            if "feat_in" not in aux_decoder_cfg and d_model:
                aux_decoder_cfg["feat_in"] = d_model
                fixes_applied.append(f"aux_ctc.decoder.feat_in = {d_model}")

            # Add vocabulary if missing
            if "vocabulary" not in aux_decoder_cfg:
                labels = config.get("labels", [])
                if labels:
                    aux_decoder_cfg["vocabulary"] = labels
                    fixes_applied.append("aux_ctc.decoder.vocabulary = [copied from labels]")

            # Update config
            aux_ctc["decoder"] = aux_decoder_cfg
            config["aux_ctc"] = aux_ctc

    # ===== Fix 3: Handle models without aux_ctc but with CTC config elsewhere =====
    # Some models might have CTC configured under different paths
    if "ctc" in config and isinstance(config["ctc"], dict):
        ctc_cfg = config["ctc"]
        if "decoder" in ctc_cfg:
            ctc_decoder = ctc_cfg["decoder"]
            if isinstance(ctc_decoder, dict) and "num_classes" not in ctc_decoder:
                if vocab_size is not None:
                    ctc_decoder["num_classes"] = vocab_size
                    config["ctc"]["decoder"] = ctc_decoder
                    fixes_applied.append(f"ctc.decoder.num_classes = {vocab_size}")

    return config, fixes_applied


def strip_ctc_from_config(config: dict) -> Tuple[dict, List[str]]:
    """
    Remove CTC-related configuration sections to convert Hybrid model to pure RNNT.

    This removes:
    - aux_ctc section (CTC decoder config)
    - interctc section (intermediate CTC loss config)

    Args:
        config: Model configuration dictionary

    Returns:
        Tuple of (modified config, list of removed sections)
    """
    removed_sections = []

    # Remove aux_ctc section
    if "aux_ctc" in config:
        del config["aux_ctc"]
        removed_sections.append("aux_ctc (CTC decoder configuration)")

    # Remove interctc section
    if "interctc" in config:
        del config["interctc"]
        removed_sections.append("interctc (intermediate CTC loss configuration)")

    return config, removed_sections


def strip_ctc_from_checkpoint(ckpt_path: Path) -> Tuple[int, int, List[str]]:
    """
    Remove CTC decoder weights from model checkpoint.

    This removes all keys starting with 'ctc_decoder.' from the state dict.

    Args:
        ckpt_path: Path to model_weights.ckpt file

    Returns:
        Tuple of (original_key_count, new_key_count, removed_keys)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for checkpoint manipulation. Install with: pip install torch")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Get state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            is_wrapped = True
        else:
            state_dict = checkpoint
            is_wrapped = False
    else:
        state_dict = checkpoint
        is_wrapped = False

    original_count = len(state_dict)
    removed_keys = []

    # Find and remove CTC-related keys
    keys_to_remove = []
    for key in state_dict.keys():
        if key.startswith('ctc_decoder.') or key.startswith('ctc_loss.'):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del state_dict[key]
        removed_keys.append(key)

    new_count = len(state_dict)

    # Save modified checkpoint
    if is_wrapped:
        checkpoint['state_dict'] = state_dict
        torch.save(checkpoint, ckpt_path)
    else:
        torch.save(state_dict, ckpt_path)

    return original_count, new_count, removed_keys


def get_rnnt_target_class(config: dict) -> str:
    """
    Determine the appropriate pure RNNT target class based on model config.

    Args:
        config: Model configuration dictionary

    Returns:
        Target class path for pure RNNT model
    """
    encoder_cfg = config.get("encoder", {})
    encoder_class = encoder_cfg.get("_target_", "") or encoder_cfg.get("target", "")

    is_fastconformer = "fastconformer" in encoder_class.lower() or "fast_conformer" in encoder_class.lower()

    # Check tokenizer type (BPE vs Char)
    tokenizer_cfg = config.get("tokenizer", {})
    has_tokenizer = bool(tokenizer_cfg)

    if has_tokenizer:
        # BPE model
        if is_fastconformer:
            return "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
        else:
            return "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
    else:
        # Char model
        return "nemo.collections.asr.models.rnnt_models.EncDecRNNTModel"


def detect_model_type(config: dict) -> str:
    """Detect architecture type from model configuration."""

    # Check encoder configuration
    encoder_cfg = config.get("encoder", {})
    encoder_class = encoder_cfg.get("_target_", "") or encoder_cfg.get("target", "")

    is_fastconformer = "fastconformer" in encoder_class.lower() or "fast_conformer" in encoder_class.lower()
    is_conformer = "conformer" in encoder_class.lower()

    # Check decoder type
    has_ctc = "aux_ctc" in config or "ctc" in str(config.get("decoding", {})).lower()
    has_rnnt = "joint" in config or "rnnt" in str(config.get("decoding", {})).lower()
    is_hybrid = has_ctc and has_rnnt

    # Check tokenizer type (BPE vs Char)
    tokenizer_cfg = config.get("tokenizer", {})
    tokenizer_dir = tokenizer_cfg.get("dir", "") or tokenizer_cfg.get("tokenizer_model", "")
    is_bpe = "bpe" in tokenizer_dir.lower() or tokenizer_cfg.get("type", "").lower() == "bpe"

    # Also check if tokenizer files exist with .model extension (SentencePiece = BPE)
    # If vocab files are present, likely BPE
    if not is_bpe:
        # Check for common BPE indicators
        vocab_cfg = config.get("vocab", {})
        if vocab_cfg or "sentencepiece" in str(tokenizer_cfg).lower():
            is_bpe = True

    print(f"  - Encoder: {'FastConformer' if is_fastconformer else 'Conformer' if is_conformer else 'Unknown'}")
    print(f"  - Decoder: {'Hybrid (RNNT+CTC)' if is_hybrid else 'RNNT' if has_rnnt else 'CTC' if has_ctc else 'Unknown'}")
    print(f"  - Tokenizer: {'BPE' if is_bpe else 'Char (assumed)'}")

    # Determine type
    if is_fastconformer or is_conformer:
        prefix = "fastconformer" if is_fastconformer else "conformer"
        if is_hybrid:
            return f"{prefix}_hybrid_bpe" if is_bpe else f"{prefix}_hybrid_char"
        elif has_rnnt:
            return f"{prefix}_rnnt_bpe" if is_bpe else f"{prefix}_rnnt_char"
        else:  # CTC
            return f"{prefix}_ctc_bpe" if is_bpe else f"{prefix}_ctc_char"

    # Default: FastConformer Hybrid BPE (most common streaming model)
    print("  - Warning: Could not determine exact model type, defaulting to fastconformer_hybrid_bpe")
    return "fastconformer_hybrid_bpe"


def diagnose_nemo_file(input_path: str) -> None:
    """
    Diagnose .nemo file config structure without making changes.

    Args:
        input_path: Path to .nemo file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"{'='*60}")
    print(f"DIAGNOSING: {input_path}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract .nemo file
        print("\n[1] Extracting .nemo archive...")
        with tarfile.open(input_path, "r") as tar:
            tar.extractall(tmpdir)

        # List files
        files = [f.name for f in tmpdir.rglob("*") if f.is_file()]
        print(f"  Files in archive: {files}")

        # Read config
        config_path = tmpdir / "model_config.yaml"
        if not config_path.exists():
            print("  ERROR: model_config.yaml not found!")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Show diagnosis
        print("\n[2] Config structure diagnosis:")
        for diag in diagnose_config_structure(config):
            print(f"  {diag}")

        # Show vocab_size detection
        print("\n[3] Vocab size detection:")
        vocab_size = get_vocab_size_from_config(config)
        print(f"  Detected vocab_size: {vocab_size}")

        # Show target
        print("\n[4] Target class:")
        print(f"  Current target: {config.get('target', 'NOT SET')}")

        # Model type detection
        print("\n[5] Detected model type:")
        model_type = detect_model_type(config)
        print(f"  Type: {model_type}")
        print(f"  Recommended target: {MODEL_TARGET_MAP.get(model_type, 'UNKNOWN')}")

        # Check for common issues
        print("\n[6] Potential issues:")
        issues = []

        # Check aux_ctc.decoder.num_classes
        aux_ctc = config.get("aux_ctc", {})
        if aux_ctc:
            aux_decoder = aux_ctc.get("decoder", {}) if isinstance(aux_ctc, dict) else {}
            if not aux_decoder:
                issues.append("aux_ctc exists but aux_ctc.decoder is missing or empty")
            elif "num_classes" not in aux_decoder:
                issues.append("aux_ctc.decoder exists but num_classes is missing")
            elif aux_decoder.get("num_classes", 0) <= 0:
                issues.append(f"aux_ctc.decoder.num_classes has invalid value: {aux_decoder.get('num_classes')}")

        # Check main decoder for CTC models
        decoder_cfg = config.get("decoder", {})
        if decoder_cfg and isinstance(decoder_cfg, dict):
            decoder_target = decoder_cfg.get("_target_", "")
            if "ctc" in decoder_target.lower() or "convasr" in decoder_target.lower():
                if "num_classes" not in decoder_cfg:
                    issues.append("CTC-style decoder but num_classes is missing")

        if issues:
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✅ No obvious issues detected")

        # Recommendations
        print("\n[7] Recommendations:")
        if aux_ctc:
            if "decoder" not in aux_ctc or not aux_ctc.get("decoder", {}).get("num_classes"):
                print("  - Run: python fix_nemo_model.py your_model.nemo --verbose")
                print("    This will add missing num_classes to aux_ctc.decoder")
            print("  - Or run: python fix_nemo_model.py your_model.nemo --strip-ctc")
            print("    This will convert Hybrid model to pure RNNT (removes CTC entirely)")
        else:
            print("  - No aux_ctc section found. This might be a pure RNNT or CTC model.")
            print("  - Run: python fix_nemo_model.py your_model.nemo --verbose")

        print(f"\n{'='*60}")


def fix_nemo_file(input_path: str, output_path: str = None, target_override: str = None,
                  strip_ctc: bool = False, verbose: bool = False) -> str:
    """
    Fix .nemo file's model_config.yaml for compatibility.

    This function:
    1. Extracts the .nemo archive
    2. Modifies model_config.yaml to fix target field
    3. Removes incompatible parameters (like sdpa_gate)
    4. [Optional] Strip CTC components for RNNT-only inference
    5. Repacks the .nemo archive

    Args:
        input_path: Input .nemo file path
        output_path: Output .nemo file path (auto-generated if None)
        target_override: Directly specify target class path (auto-detect if None)
        strip_ctc: If True, remove CTC decoder and convert to pure RNNT model

    Returns:
        Path to the fixed .nemo file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.suffix == ".nemo":
        raise ValueError(f"Input file is not a .nemo file: {input_path}")

    if output_path is None:
        suffix = "_rnnt" if strip_ctc else "_fixed"
        output_path = input_path.parent / f"{input_path.stem}{suffix}.nemo"
    else:
        output_path = Path(output_path)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    if strip_ctc:
        print(f"Mode: Strip CTC (convert Hybrid -> pure RNNT)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Extract .nemo file (tarball)
        print("\n[1/6] Extracting .nemo archive...")
        with tarfile.open(input_path, "r") as tar:
            tar.extractall(tmpdir)

        # List extracted files
        extracted_files = list(tmpdir.rglob("*"))
        print(f"  - Extracted files: {[f.name for f in extracted_files if f.is_file()]}")

        # 2. Read model_config.yaml
        print("\n[2/6] Analyzing model_config.yaml...")
        config_path = tmpdir / "model_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.yaml not found in archive")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Show current target
        current_target = config.get("target", "None")
        print(f"  - Current target: {current_target}")

        # Determine total steps based on mode
        total_steps = 7 if strip_ctc else 6
        step = 2

        # 3. Fix target field
        step += 1
        print(f"\n[{step}/{total_steps}] Fixing target field...")

        if strip_ctc:
            # Force pure RNNT target when stripping CTC
            new_target = get_rnnt_target_class(config)
            print(f"  - Converting to pure RNNT model")
            print(f"  - New target: {new_target}")
        elif target_override:
            new_target = target_override
            print(f"  - Using user-specified target: {new_target}")
        else:
            # Auto-detect model type
            model_type = detect_model_type(config)
            new_target = MODEL_TARGET_MAP.get(model_type)

            if new_target is None:
                raise ValueError(f"Unsupported model type: {model_type}")

            print(f"  - Detected model type: {model_type}")
            print(f"  - New target: {new_target}")

        # Update target field
        config["target"] = new_target

        # 4. Remove incompatible parameters
        step += 1
        print(f"\n[{step}/{total_steps}] Removing incompatible parameters...")
        config, removed_params = remove_incompatible_params(config)

        if removed_params:
            print(f"  - Removed parameters:")
            for param in removed_params:
                print(f"    * {param}")
        else:
            print(f"  - No incompatible parameters found")

        # Variables to track what was done
        hybrid_fixes = []
        ctc_config_removed = []
        ctc_weights_removed = []

        if strip_ctc:
            # 5. Strip CTC from config
            step += 1
            print(f"\n[{step}/{total_steps}] Stripping CTC configuration...")
            config, ctc_config_removed = strip_ctc_from_config(config)

            if ctc_config_removed:
                print(f"  - Removed config sections:")
                for section in ctc_config_removed:
                    print(f"    * {section}")
            else:
                print(f"  - No CTC configuration found to remove")

            # 6. Strip CTC from checkpoint
            step += 1
            print(f"\n[{step}/{total_steps}] Stripping CTC weights from checkpoint...")
            ckpt_path = tmpdir / "model_weights.ckpt"

            if ckpt_path.exists():
                orig_count, new_count, ctc_weights_removed = strip_ctc_from_checkpoint(ckpt_path)
                print(f"  - Original weight keys: {orig_count}")
                print(f"  - After stripping: {new_count}")
                print(f"  - Removed {len(ctc_weights_removed)} CTC-related weight keys")

                if ctc_weights_removed:
                    # Show a few example keys
                    for key in ctc_weights_removed[:5]:
                        print(f"    * {key}")
                    if len(ctc_weights_removed) > 5:
                        print(f"    * ... and {len(ctc_weights_removed) - 5} more")
            else:
                print(f"  - WARNING: model_weights.ckpt not found, skipping weight stripping")
        else:
            # 5. Fix Hybrid model config (add num_classes to aux_ctc.decoder, etc.)
            step += 1
            print(f"\n[{step}/{total_steps}] Fixing Hybrid RNNT+CTC model config...")
            config, hybrid_fixes = fix_hybrid_model_config(config, verbose=verbose)

            if hybrid_fixes:
                print(f"  - Applied fixes:")
                for fix in hybrid_fixes:
                    print(f"    * {fix}")
            else:
                print(f"  - No Hybrid model fixes needed (not a Hybrid model or already configured)")

        # Save modified config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print("  - model_config.yaml saved")

        # Final step: Repack as new .nemo file
        step += 1
        print(f"\n[{step}/{total_steps}] Creating new .nemo archive...")

        with tarfile.open(output_path, "w") as tar:
            for file_path in tmpdir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmpdir)
                    tar.add(file_path, arcname=arcname)
                    print(f"  - Added: {arcname}")

        print(f"\n{'='*60}")
        if strip_ctc:
            print(f"SUCCESS! Converted to pure RNNT model: {output_path}")
        else:
            print(f"SUCCESS! Fixed model saved to: {output_path}")
        print(f"{'='*60}")

        if removed_params:
            print(f"\nNote: The following parameters were removed for compatibility:")
            for param in removed_params:
                print(f"  - {param}")

        if hybrid_fixes:
            print(f"\nNote: The following Hybrid model fixes were applied:")
            for fix in hybrid_fixes:
                print(f"  - {fix}")

        if ctc_config_removed:
            print(f"\nNote: The following CTC config sections were removed:")
            for section in ctc_config_removed:
                print(f"  - {section}")

        if ctc_weights_removed:
            print(f"\nNote: Removed {len(ctc_weights_removed)} CTC weight keys from checkpoint")

        # Calculate and show file size reduction if stripping CTC
        if strip_ctc:
            input_size = os.path.getsize(input_path) / (1024 * 1024)
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            reduction = input_size - output_size
            print(f"\nFile size comparison:")
            print(f"  - Input:  {input_size:.1f} MB")
            print(f"  - Output: {output_size:.1f} MB")
            print(f"  - Saved:  {reduction:.1f} MB ({reduction/input_size*100:.1f}% reduction)")

        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fix .nemo model files for compatibility with older NeMo versions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect mode (recommended)
  python fix_nemo_model.py my_model.nemo

  # Specify output file
  python fix_nemo_model.py my_model.nemo my_model_fixed.nemo

  # Directly specify target class
  python fix_nemo_model.py my_model.nemo --target nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel

Supported model types:
  - fastconformer_hybrid_bpe (FastConformer Hybrid RNNT+CTC BPE)
  - fastconformer_hybrid_char (FastConformer Hybrid RNNT+CTC Char)
  - fastconformer_ctc_bpe (FastConformer CTC BPE)
  - fastconformer_rnnt_bpe (FastConformer RNNT BPE)
  - conformer_hybrid_bpe (Conformer Hybrid RNNT+CTC BPE)
  - conformer_hybrid_char (Conformer Hybrid RNNT+CTC Char)
  - conformer_ctc_bpe (Conformer CTC BPE)
  - conformer_rnnt_bpe (Conformer RNNT BPE)
  - conformer_ctc_char (Conformer CTC Char)
  - conformer_rnnt_char (Conformer RNNT Char)

This script also:
- Removes incompatible parameters like 'sdpa_gate' that exist in newer
  NeMo versions but cause errors in older versions
- Adds missing 'num_classes' to aux_ctc.decoder for Hybrid RNNT+CTC models
- Sets 'feat_in' for CTC decoder from encoder.d_model if missing

Strip CTC mode (--strip-ctc):
  Converts Hybrid RNNT+CTC model to pure RNNT model by:
  - Removing aux_ctc and interctc config sections
  - Removing ctc_decoder.* weights from checkpoint
  - Changing target class to EncDecRNNTBPEModel

  This is useful when:
  - You only use RNNT decoding (not CTC)
  - You want to reduce model file size
  - You want to reduce memory footprint during inference

  Example:
    python fix_nemo_model.py hybrid_model.nemo --strip-ctc
    # Output: hybrid_model_rnnt.nemo
        """
    )

    parser.add_argument("input", help="Input .nemo file path")
    parser.add_argument("output", nargs="?", help="Output .nemo file path (default: input_fixed.nemo or input_rnnt.nemo)")
    parser.add_argument("--target", "-t", help="Directly specify target class path")
    parser.add_argument("--strip-ctc", "-s", action="store_true",
                        help="Strip CTC decoder and convert Hybrid model to pure RNNT (reduces model size)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose mode with detailed config diagnostics")
    parser.add_argument("--diagnose", "-d", action="store_true",
                        help="Only diagnose config structure without making changes")
    parser.add_argument("--list-targets", "-l", action="store_true", help="List supported target classes")

    args = parser.parse_args()

    if args.list_targets:
        print("Supported model types and target classes:")
        print("-" * 60)
        for model_type, target in MODEL_TARGET_MAP.items():
            print(f"  {model_type}:")
            print(f"    -> {target}")
            print()
        return

    if args.diagnose:
        try:
            diagnose_nemo_file(args.input)
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        return

    try:
        fix_nemo_file(args.input, args.output, args.target,
                      strip_ctc=args.strip_ctc, verbose=args.verbose)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
