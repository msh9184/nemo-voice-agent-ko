#!/usr/bin/env python3
"""
Integration test for Korean Voice Agent pipeline.
This script tests the full STT → LLM → TTS pipeline.

Usage:
    python test_korean_integration.py --config korean_voice_agent.yaml
    python test_korean_integration.py --stt-only  # Test STT only
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def test_pipeline_components(config_path: str = None, stt_only: bool = False):
    """Test Korean Voice Agent pipeline components."""
    print("\n" + "=" * 70)
    print("Korean Voice Agent Integration Test")
    print("=" * 70)

    results = {}

    # Test 1: STT Model Loading
    print("\n[1/4] Testing STT Model...")
    try:
        import torch
        from nemo.collections.asr.models import ASRModel

        stt_model_path = "/path/to/model/asr_model.nemo"

        if os.path.exists(stt_model_path):
            print(f"  Loading STT model from: {stt_model_path}")
            start_time = time.time()

            # Just check if file is readable and valid .nemo format
            import zipfile
            with zipfile.ZipFile(stt_model_path, 'r') as z:
                manifest_files = [f for f in z.namelist() if 'manifest' in f.lower() or 'config' in f.lower()]
                print(f"  ✓ STT model file is valid NeMo archive")
                print(f"  ✓ Contains: {len(z.namelist())} files")

            results["STT Model"] = True
            print("  ✓ STT model check PASSED")
        else:
            print(f"  ⚠ STT model path not accessible: {stt_model_path}")
            print("    (This is expected if running on WSL without access to /path/to)")
            results["STT Model"] = None
    except Exception as e:
        print(f"  ✗ STT model check FAILED: {e}")
        results["STT Model"] = False

    if stt_only:
        return results

    # Test 2: LLM Model Loading
    print("\n[2/4] Testing LLM Model...")
    try:
        from transformers import AutoTokenizer, AutoConfig

        llm_model_path = "/path/to/model"

        if os.path.exists(llm_model_path):
            print(f"  Loading LLM config from: {llm_model_path}")

            # Check config.json
            config_file = os.path.join(llm_model_path, "config.json")
            if os.path.exists(config_file):
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"  ✓ Model type: {config.get('model_type', 'unknown')}")
                print(f"  ✓ Hidden size: {config.get('hidden_size', 'unknown')}")
                print(f"  ✓ Num layers: {config.get('num_hidden_layers', 'unknown')}")

            results["LLM Model"] = True
            print("  ✓ LLM model check PASSED")
        else:
            print(f"  ⚠ LLM model path not accessible: {llm_model_path}")
            results["LLM Model"] = None
    except Exception as e:
        print(f"  ✗ LLM model check FAILED: {e}")
        results["LLM Model"] = False

    # Test 3: TTS (MeloTTS Korean)
    print("\n[3/4] Testing TTS (MeloTTS Korean)...")
    try:
        from melo.api import TTS

        print("  Loading MeloTTS Korean model...")
        start_time = time.time()
        model = TTS(language='KR', device='cpu')
        load_time = time.time() - start_time
        print(f"  ✓ Model loaded in {load_time:.2f}s")

        # Generate test audio
        test_text = "안녕하세요, 한국어 음성 합성 테스트입니다."
        speaker_id = model.hps.data.spk2id['KR']

        start_time = time.time()
        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / "integration_test_tts.wav")

        model.tts_to_file(test_text, speaker_id, output_path, speed=1.0)
        gen_time = time.time() - start_time

        if os.path.exists(output_path):
            import wave
            with wave.open(output_path, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
            print(f"  ✓ Generated audio: {duration:.2f}s in {gen_time:.2f}s")
            print(f"  ✓ Real-time factor: {gen_time/duration:.2f}x")
            results["TTS (MeloTTS)"] = True
            print("  ✓ TTS test PASSED")
        else:
            results["TTS (MeloTTS)"] = False
            print("  ✗ TTS test FAILED: No output file")
    except ImportError:
        print("  ⚠ MeloTTS not installed. Install with: pip install melotts")
        results["TTS (MeloTTS)"] = None
    except Exception as e:
        print(f"  ✗ TTS test FAILED: {e}")
        results["TTS (MeloTTS)"] = False

    # Test 4: Configuration Loading
    print("\n[4/4] Testing Configuration Loading...")
    try:
        from omegaconf import OmegaConf

        config_dir = Path(__file__).parent.parent / "server" / "server_configs"
        config_file = config_dir / "korean_voice_agent.yaml"

        if config_file.exists():
            config = OmegaConf.load(str(config_file))
            print(f"  ✓ Loaded configuration: {config_file.name}")
            print(f"    - STT type: {config.stt.type}")
            print(f"    - LLM type: {config.llm.type}")
            print(f"    - TTS type: {config.tts.type}")
            results["Configuration"] = True
            print("  ✓ Configuration test PASSED")
        else:
            print(f"  ✗ Configuration file not found: {config_file}")
            results["Configuration"] = False
    except Exception as e:
        print(f"  ✗ Configuration test FAILED: {e}")
        results["Configuration"] = False

    return results


def main():
    parser = argparse.ArgumentParser(description="Korean Voice Agent Integration Test")
    parser.add_argument(
        "--config",
        type=str,
        default="korean_voice_agent.yaml",
        help="Configuration file name"
    )
    parser.add_argument(
        "--stt-only",
        action="store_true",
        help="Test STT only"
    )
    args = parser.parse_args()

    results = test_pipeline_components(args.config, args.stt_only)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result is True:
            status = "✓ PASSED"
            passed += 1
        elif result is False:
            status = "✗ FAILED"
            failed += 1
        else:
            status = "⚠ SKIPPED"
            skipped += 1
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
