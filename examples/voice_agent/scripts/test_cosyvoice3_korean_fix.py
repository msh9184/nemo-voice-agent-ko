#!/usr/bin/env python3
"""
Test script for CosyVoice3 Korean TTS with correct parameters.

Key fix: Use text_frontend=False for Korean language support.

The text frontend in CosyVoice only supports Chinese and English normalization.
For Korean (and other languages), text_frontend must be disabled.

Reference: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
"NOTE if you want to reproduce the results, please add text_frontend=False during inference"

Usage:
    python test_cosyvoice3_korean_fix.py --model /path/to/Fun-CosyVoice3-0.5B
    python test_cosyvoice3_korean_fix.py --reference /path/to/korean_reference.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def test_cosyvoice3_korean(model_path: str, reference_audio: str = None):
    """Test CosyVoice3 Korean TTS with text_frontend=False."""
    print("\n" + "=" * 70)
    print("CosyVoice3 Korean TTS Test (with text_frontend=False)")
    print("=" * 70)

    # Check model path
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return False

    print(f"\nModel path: {model_path}")
    if reference_audio:
        print(f"Reference audio: {reference_audio}")

    # Import PyTorch
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available (will be slow)")
    except ImportError as e:
        print(f"PyTorch import error: {e}")
        return False

    # Load CosyVoice model
    print("\n" + "-" * 70)
    print("Loading CosyVoice3 model...")
    print("-" * 70)

    model = None
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        start_time = time.time()
        model = CosyVoice2(model_path)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    if model is None:
        print("Model is None")
        return False

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {gpu_memory:.2f} GB")

    # Get sample rate
    sample_rate = getattr(model, 'sample_rate', 24000)
    print(f"Sample rate: {sample_rate}")

    # Output directory
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Load reference audio if provided
    prompt_speech = None
    if reference_audio and os.path.exists(reference_audio):
        try:
            from cosyvoice.utils.file_utils import load_wav
            # CosyVoice expects 16kHz for prompt audio
            prompt_speech = load_wav(reference_audio, 16000)
            print(f"Loaded reference audio: {reference_audio}")
            print(f"Reference audio shape: {prompt_speech.shape}")
        except Exception as e:
            print(f"Failed to load reference audio: {e}")

    # Korean test texts with language tag
    test_cases = [
        # Test 1: Simple greeting with Korean tag
        {
            "name": "korean_simple",
            "text": "<|ko|>안녕하세요",
            "method": "inference_cross_lingual",
            "text_frontend": False,  # KEY FIX!
        },
        # Test 2: Korean sentence with tag
        {
            "name": "korean_sentence",
            "text": "<|ko|>반갑습니다. 저는 인공지능 음성 비서입니다.",
            "method": "inference_cross_lingual",
            "text_frontend": False,
        },
        # Test 3: Longer Korean text
        {
            "name": "korean_long",
            "text": "<|ko|>오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날씨입니다.",
            "method": "inference_cross_lingual",
            "text_frontend": False,
        },
        # Test 4: Without language tag (for comparison)
        {
            "name": "korean_no_tag",
            "text": "안녕하세요. 한국어 테스트입니다.",
            "method": "inference_cross_lingual",
            "text_frontend": False,
        },
        # Test 5: Using inference_instruct2 with Korean instruction
        {
            "name": "korean_instruct",
            "text": "안녕하세요. 반갑습니다.",
            "instruct": "Speak in Korean naturally",
            "method": "inference_instruct2",
            "text_frontend": False,
        },
        # Test 6: Using inference_instruct2 with Chinese instruction (for comparison)
        {
            "name": "korean_instruct_zh",
            "text": "안녕하세요. 반갑습니다.",
            "instruct": "用韩语自然地说",
            "method": "inference_instruct2",
            "text_frontend": False,
        },
    ]

    print("\n" + "-" * 70)
    print("Testing Korean speech synthesis with text_frontend=False")
    print("-" * 70)

    results = []
    for i, test in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test['name']}")
        print(f"  Text: {test['text']}")
        print(f"  Method: {test['method']}")
        print(f"  text_frontend: {test['text_frontend']}")
        if "instruct" in test:
            print(f"  Instruct: {test['instruct']}")

        try:
            start_time = time.time()
            audio_chunks = []

            if test["method"] == "inference_cross_lingual":
                if prompt_speech is None:
                    print("  SKIP: No reference audio for cross_lingual")
                    results.append({"name": test["name"], "success": False, "reason": "no_reference"})
                    continue

                # KEY FIX: text_frontend=False
                generator = model.inference_cross_lingual(
                    test["text"],
                    prompt_speech,
                    stream=False,
                    text_frontend=test["text_frontend"]  # IMPORTANT!
                )

            elif test["method"] == "inference_instruct2":
                if prompt_speech is None:
                    print("  SKIP: No reference audio for instruct2")
                    results.append({"name": test["name"], "success": False, "reason": "no_reference"})
                    continue

                # KEY FIX: text_frontend=False
                generator = model.inference_instruct2(
                    test["text"],
                    test["instruct"],
                    prompt_speech,
                    stream=False,
                    text_frontend=test["text_frontend"]  # IMPORTANT!
                )

            else:
                print(f"  Unknown method: {test['method']}")
                results.append({"name": test["name"], "success": False, "reason": "unknown_method"})
                continue

            # Collect audio chunks
            for chunk in generator:
                if isinstance(chunk, dict) and 'tts_speech' in chunk:
                    audio_data = chunk['tts_speech']
                else:
                    audio_data = chunk

                if hasattr(audio_data, 'cpu'):
                    audio_data = audio_data.cpu().numpy()
                elif hasattr(audio_data, 'numpy'):
                    audio_data = audio_data.numpy()

                audio_chunks.append(audio_data)

            gen_time = time.time() - start_time

            if audio_chunks:
                # Concatenate and save
                audio = np.concatenate([c.flatten() for c in audio_chunks])
                duration = len(audio) / sample_rate

                print(f"  Generated: {duration:.2f}s audio in {gen_time:.2f}s")
                print(f"  RTF: {gen_time/duration:.2f}x")

                # Save audio
                output_path = output_dir / f"cosyvoice3_fix_{test['name']}.wav"
                try:
                    import scipy.io.wavfile as wavfile
                    # Normalize to int16
                    if audio.dtype == np.float32 or audio.dtype == np.float64:
                        audio_int16 = (audio * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio.astype(np.int16)
                    wavfile.write(str(output_path), sample_rate, audio_int16)
                    print(f"  Saved: {output_path}")
                except Exception as e:
                    print(f"  Save failed: {e}")

                results.append({
                    "name": test["name"],
                    "success": True,
                    "duration": duration,
                    "gen_time": gen_time,
                    "rtf": gen_time / duration
                })
            else:
                print("  No audio generated")
                results.append({"name": test["name"], "success": False, "reason": "no_audio"})

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": test["name"], "success": False, "reason": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        print(f"\nAverage RTF: {avg_rtf:.2f}x")

    print(f"\nOutput directory: {output_dir}")
    print("\nPLEASE CHECK THE GENERATED AUDIO FILES TO VERIFY KOREAN OUTPUT!")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Test CosyVoice3 Korean TTS (fixed)")
    parser.add_argument(
        "--model",
        type=str,
        default="/path/to/model",
        help="Path to CosyVoice3 model"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="/path/to/workspace",
        help="Path to reference audio (3-10 seconds of speech)"
    )
    args = parser.parse_args()

    success = test_cosyvoice3_korean(args.model, args.reference)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
