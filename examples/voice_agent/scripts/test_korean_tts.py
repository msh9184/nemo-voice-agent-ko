#!/usr/bin/env python3
"""
Test script for Korean TTS services.
This script tests MeloTTS Korean and Kokoro Korean TTS implementations.

Usage:
    python test_korean_tts.py --tts melo
    python test_korean_tts.py --tts kokoro
    python test_korean_tts.py --tts all
"""

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def test_melo_tts():
    """Test MeloTTS Korean service."""
    print("\n" + "=" * 60)
    print("Testing MeloTTS Korean TTS")
    print("=" * 60)

    try:
        from melo.api import TTS

        print("✓ MeloTTS package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MeloTTS: {e}")
        print("  Install with: pip install melotts")
        return False

    test_texts = [
        "안녕하세요! 오늘은 날씨가 정말 좋네요.",
        "저는 한국어 음성 합성 시스템입니다.",
        "무엇을 도와드릴까요?",
    ]

    try:
        print("\nLoading MeloTTS Korean model...")
        start_time = time.time()
        model = TTS(language='KR', device='cpu')
        speaker_id = model.hps.data.spk2id['KR']
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        print(f"  Sample rate: {model.hps.data.sampling_rate}")
        print(f"  Speaker ID: {speaker_id}")

        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)

        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: '{text}'")
            start_time = time.time()

            # Generate audio
            output_path = str(output_dir / f"melo_test_{i+1}.wav")
            model.tts_to_file(text, speaker_id, output_path, speed=1.0)

            gen_time = time.time() - start_time

            # Check output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                with wave.open(output_path, 'rb') as wf:
                    duration = wf.getnframes() / wf.getframerate()
                print(f"  ✓ Generated in {gen_time:.2f}s")
                print(f"  ✓ Duration: {duration:.2f}s, Size: {file_size} bytes")
                print(f"  ✓ Real-time factor: {gen_time/duration:.2f}x")
            else:
                print(f"  ✗ Output file not created")
                return False

        print("\n✓ MeloTTS Korean test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ MeloTTS Korean test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kokoro_korean():
    """Test Kokoro Korean TTS service."""
    print("\n" + "=" * 60)
    print("Testing Kokoro Korean TTS")
    print("=" * 60)

    try:
        from kokoro import KPipeline

        print("✓ Kokoro package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Kokoro: {e}")
        print("  Install with: pip install kokoro>=0.9.2")
        return False

    test_texts = [
        "안녕하세요! 코코로 한국어 음성 합성입니다.",
        "오늘 하루도 좋은 하루 되세요.",
    ]

    try:
        print("\nLoading Kokoro Korean model (lang_code='k')...")
        start_time = time.time()
        pipeline = KPipeline(lang_code='k')
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")

        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)

        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: '{text}'")
            start_time = time.time()

            # Generate audio
            audio_chunks = []
            for gs, ps, audio in pipeline(text, voice='kf_yujeong', speed=1.0):
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                audio_chunks.append(audio)

            gen_time = time.time() - start_time

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                duration = len(full_audio) / 24000  # Kokoro uses 24kHz

                # Save to file
                output_path = str(output_dir / f"kokoro_korean_test_{i+1}.wav")
                audio_int16 = (full_audio * 32767).astype(np.int16)
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_int16.tobytes())

                print(f"  ✓ Generated in {gen_time:.2f}s")
                print(f"  ✓ Duration: {duration:.2f}s")
                print(f"  ✓ Real-time factor: {gen_time/duration:.2f}x")
            else:
                print(f"  ✗ No audio generated")
                return False

        print("\n✓ Kokoro Korean test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Kokoro Korean test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Korean TTS services")
    parser.add_argument(
        "--tts",
        choices=["melo", "kokoro", "all"],
        default="all",
        help="TTS service to test"
    )
    args = parser.parse_args()

    results = {}

    if args.tts in ["melo", "all"]:
        results["MeloTTS Korean"] = test_melo_tts()

    if args.tts in ["kokoro", "all"]:
        results["Kokoro Korean"] = test_kokoro_korean()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    # Exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
