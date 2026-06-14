#!/usr/bin/env python3
"""
Comprehensive Test Script for CosyVoice3 Korean TTS

This script tests CosyVoice3 Korean TTS with ALL required fixes:
1. text_frontend=False - REQUIRED for Korean (text frontend only supports Chinese/English)
2. prompt_text - REQUIRED: The EXACT transcript of the reference audio
3. prompt_wav - Reference audio (parameter name confirmed from official source code)

Based on:
- Official HuggingFace docs: https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B-2512
- Official GitHub: https://github.com/FunAudioLLM/CosyVoice
- CosyVoice3 Paper: https://arxiv.org/html/2505.17589v1

Usage:
    python test_cosyvoice3_korean_complete.py \
        --model /path/to/model \
        --reference /path/to/korean_reference.wav \
        --reference_text "참조 오디오가 말하는 정확한 내용"

Note: The reference_text MUST match exactly what is spoken in the reference audio!
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def test_cosyvoice3_korean(
    model_path: str,
    reference_audio: str = None,
    reference_text: str = None
):
    """
    Test CosyVoice3 Korean TTS with all required parameters.

    Args:
        model_path: Path to CosyVoice3 model directory
        reference_audio: Path to Korean reference audio (3-10 seconds)
        reference_text: The EXACT transcript of the reference audio (CRITICAL!)
    """
    print("\n" + "=" * 80)
    print("CosyVoice3 Korean TTS Complete Test")
    print("=" * 80)
    print("\nCRITICAL REQUIREMENTS for Korean TTS:")
    print("  1. text_frontend=False (text frontend only supports Chinese/English)")
    print("  2. prompt_text = exact transcript of reference audio")
    print("  3. prompt_wav = file path to reference audio (CosyVoice loads internally at 24kHz)")
    print("=" * 80)

    # Validate inputs
    if not os.path.exists(model_path):
        print(f"\nERROR: Model path does not exist: {model_path}")
        return False

    print(f"\nModel path: {model_path}")
    print(f"Reference audio: {reference_audio}")
    print(f"Reference text: '{reference_text}'")

    if reference_audio and not reference_text:
        print("\nWARNING: reference_text not provided!")
        print("For best Korean TTS quality, you MUST provide the exact transcript of the reference audio.")

    # Import PyTorch
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available - will use CPU (slower)")
    except ImportError as e:
        print(f"PyTorch import error: {e}")
        return False

    # Load CosyVoice model
    print("\n" + "-" * 80)
    print("Loading CosyVoice3 model...")
    print("-" * 80)

    model = None
    try:
        # Try CosyVoice3 class (used by CosyVoice3)
        from cosyvoice.cli.cosyvoice import CosyVoice3
        start_time = time.time()
        model = CosyVoice3(model_path)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s using CosyVoice3 API")
    except ImportError:
        try:
            from cosyvoice.cli.model import AutoModel
            start_time = time.time()
            model = AutoModel(model_dir=model_path)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s using AutoModel API")
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

    # Validate reference audio path (CosyVoice loads it internally)
    # IMPORTANT: prompt_wav expects FILE PATH (string), not pre-loaded tensor!
    # CosyVoice internally calls load_wav(prompt_wav, 24000) to load the audio
    reference_audio_valid = False
    if reference_audio and os.path.exists(reference_audio):
        reference_audio_valid = True
        print(f"\nReference audio path: {reference_audio}")

        # Just check file size to estimate duration (don't load it ourselves)
        file_size = os.path.getsize(reference_audio)
        # Rough estimate: 16kHz mono 16-bit = 32KB/sec
        estimated_duration = file_size / 32000
        print(f"Reference audio file size: {file_size / 1024:.1f} KB")
        print(f"Estimated duration: {estimated_duration:.1f}s (3-10s recommended)")
    elif reference_audio:
        print(f"\nERROR: Reference audio file not found: {reference_audio}")

    # Prepare reference text (prompt_text)
    prompt_text = reference_text or ""
    if reference_audio_valid and not prompt_text:
        print("\nWARNING: No prompt_text provided for reference audio!")
        print("This will likely result in poor voice cloning quality.")
        print("Please provide the exact transcript of the reference audio.")

    # Test cases - Now using CROSS-LINGUAL inference with <|ko|> language tag
    test_cases = [
        {
            "name": "cross_lingual_basic",
            "text": "안녕하세요",
            "description": "Basic greeting - Cross-lingual with <|ko|> tag",
        },
        {
            "name": "cross_lingual_sentence",
            "text": "반갑습니다. 저는 인공지능 음성 비서입니다.",
            "description": "Full sentence - Cross-lingual mode",
        },
        {
            "name": "cross_lingual_question",
            "text": "오늘 날씨가 어떤가요?",
            "description": "Question - Cross-lingual mode",
        },
        {
            "name": "cross_lingual_long",
            "text": "안녕하세요, 오늘 날씨가 정말 좋네요. 어떤 음악을 들려드릴까요?",
            "description": "Longer text - Cross-lingual mode",
        },
    ]

    print("\n" + "-" * 80)
    print("Testing Korean TTS with CROSS-LINGUAL inference")
    print("-" * 80)
    print("\nKEY INSIGHT: For Korean, use inference_cross_lingual with <|ko|> tag!")
    print("  - inference_zero_shot: for SAME language as reference (no language tag)")
    print("  - inference_cross_lingual: for DIFFERENT language (requires <|ko|> tag)")
    print(f"\nConfiguration:")
    print(f"  Method: inference_cross_lingual")
    print(f"  Language tag: <|ko|> (REQUIRED for Korean!)")
    print(f"  text_frontend: False (REQUIRED for Korean)")
    print(f"  prompt_wav (file path): {reference_audio if reference_audio_valid else 'None'}")
    print(f"  stream: False (for testing)")

    results = []
    for i, test in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Text: {test['text']}")

        if not reference_audio_valid:
            print("  SKIP: No valid reference audio provided (required for cross-lingual)")
            results.append({"name": test["name"], "success": False, "reason": "no_reference"})
            continue

        try:
            # CRITICAL: Use inference_cross_lingual for Korean!
            # - inference_zero_shot: SAME language as reference, NO language tag
            # - inference_cross_lingual: DIFFERENT language, REQUIRES <|ko|> tag
            tts_text_with_tag = f"<|ko|>{test['text']}"  # Add Korean language tag!
            print(f"  TTS text with language tag: {tts_text_with_tag}")

            start_time = time.time()
            audio_chunks = []

            # Use inference_cross_lingual for Korean
            # Parameters: tts_text (with tag), prompt_wav, stream, text_frontend
            # NOTE: cross_lingual does NOT take prompt_text parameter!
            print(f"  Calling inference_cross_lingual with:")
            print(f"    tts_text={tts_text_with_tag} (<|ko|> tag REQUIRED!)")
            print(f"    prompt_wav={reference_audio} (FILE PATH)")
            print(f"    text_frontend=False (REQUIRED for Korean)")

            generator = model.inference_cross_lingual(
                tts_text=tts_text_with_tag,  # Korean text WITH <|ko|> tag!
                prompt_wav=reference_audio,  # FILE PATH (string)
                stream=False,
                text_frontend=False  # REQUIRED for Korean!
            )

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

                print(f"  SUCCESS: {duration:.2f}s audio in {gen_time:.2f}s")
                print(f"  RTF: {gen_time/duration:.2f}x")

                # Save audio
                output_path = output_dir / f"cosyvoice3_complete_{test['name']}.wav"
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
                print("  FAILED: No audio generated")
                results.append({"name": test["name"], "success": False, "reason": "no_audio"})

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": test["name"], "success": False, "reason": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        print(f"\nAverage RTF: {avg_rtf:.2f}x")

    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('reason', 'unknown')}")

    print(f"\nOutput directory: {output_dir}")

    print("\n" + "=" * 80)
    print("IMPORTANT: Please listen to the generated audio files!")
    print("If you hear Korean speech (not Chinese), the fix is working.")
    print("=" * 80)

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Test CosyVoice3 Korean TTS with complete fix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python test_cosyvoice3_korean_complete.py \\
      --model /path/to/model \\
      --reference /path/to/korean_voice.wav \\
      --reference_text "안녕하세요, 저는 인공지능 비서입니다."

CRITICAL: The --reference_text must EXACTLY match what is spoken in the reference audio!
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/path/to/model",
        help="Path to CosyVoice3 model directory"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference audio (3-10 seconds of Korean speech)"
    )
    parser.add_argument(
        "--reference_text",
        type=str,
        default=None,
        help="CRITICAL: The EXACT transcript of the reference audio"
    )
    args = parser.parse_args()

    # Warn if reference_text is not provided
    if args.reference and not args.reference_text:
        print("\n" + "!" * 80)
        print("WARNING: --reference_text not provided!")
        print("For proper Korean TTS, you MUST provide the exact transcript of the reference audio.")
        print("Without it, voice cloning quality will be significantly degraded.")
        print("!" * 80 + "\n")

    success = test_cosyvoice3_korean(
        model_path=args.model,
        reference_audio=args.reference,
        reference_text=args.reference_text
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
