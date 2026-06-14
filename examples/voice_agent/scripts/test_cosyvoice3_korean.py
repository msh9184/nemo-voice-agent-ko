#!/usr/bin/env python3
"""
Test script for CosyVoice3 Korean TTS (Fun-CosyVoice3-0.5B).
This script tests the CosyVoice3 model loading and Korean speech synthesis.

Model: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

Usage:
    python test_cosyvoice3_korean.py --model /path/to/Fun-CosyVoice3-0.5B
    python test_cosyvoice3_korean.py  # Uses default path
    python test_cosyvoice3_korean.py --streaming  # Test streaming mode
    python test_cosyvoice3_korean.py --reference /path/to/reference.wav  # Zero-shot voice cloning
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def test_cosyvoice3_korean(model_path: str, streaming: bool = True, reference_audio: str = None):
    """Test CosyVoice3 Korean TTS."""
    print("\n" + "=" * 70)
    print("Testing CosyVoice3 Korean TTS (Fun-CosyVoice3-0.5B)")
    print("=" * 70)

    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        print("  Please provide the correct path to the CosyVoice3 model")
        print("  Download from: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
        return False

    print(f"\nModel path: {model_path}")
    print(f"Streaming mode: {streaming}")
    if reference_audio:
        print(f"Reference audio: {reference_audio}")

    # List model files
    print("\nModel files:")
    expected_files = ["llm.pt", "flow.pt", "hift.pt", "speech_tokenizer_v2.onnx", "campplus.onnx", "spk2info.pt"]
    found_files = []
    for f in os.listdir(model_path):
        file_path = os.path.join(model_path, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {f} ({size:.2f} MB)")
            found_files.append(f)

    # Check for expected files
    missing_files = [f for f in expected_files if f not in found_files]
    if missing_files:
        print(f"\nWarning: Missing expected files: {missing_files}")
        print("  Some features may not work properly.")

    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA not available, using CPU (will be slow)")
    except ImportError as e:
        print(f"Failed to import torch: {e}")
        return False

    # Try to import CosyVoice
    print("\n" + "-" * 70)
    print("Loading CosyVoice3 model...")
    print("-" * 70)

    model = None
    try:
        # Try the newer AutoModel API first
        try:
            from cosyvoice.cli.model import AutoModel
            print("Using AutoModel API...")
            start_time = time.time()
            model = AutoModel(model_dir=model_path)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s using AutoModel")
        except ImportError:
            # Fallback to CosyVoice2
            from cosyvoice.cli.cosyvoice import CosyVoice2
            print("Using CosyVoice2 API...")
            start_time = time.time()
            model = CosyVoice2(model_path)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s using CosyVoice2")
    except ImportError as e:
        print(f"Failed to import CosyVoice: {e}")
        print("\nPlease install CosyVoice:")
        print("  git clone https://github.com/FunAudioLLM/CosyVoice.git")
        print("  cd CosyVoice && pip install -e .")
        return False
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    if model is None:
        print("Failed to load model")
        return False

    # Check GPU memory usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used after loading: {gpu_memory:.2f} GB")

    # Test Korean TTS
    print("\n" + "-" * 70)
    print("Testing Korean speech synthesis...")
    print("-" * 70)

    test_texts = [
        "안녕하세요",
        "반갑습니다. 저는 인공지능 음성 비서입니다.",
        "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날씨입니다.",
        "한국어 음성 합성 테스트입니다. 자연스러운 억양으로 말할 수 있습니다.",
    ]

    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    results = []
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/{len(test_texts)}] Text: {text}")

        try:
            # Add Korean language tag
            text_with_tag = f"<|ko|>{text}"

            start_time = time.time()
            audio_data = None
            total_samples = 0

            if streaming:
                # Streaming mode
                print("  Generating with streaming...")
                chunks = []
                if reference_audio and os.path.exists(reference_audio):
                    # Zero-shot with reference audio
                    generator = model.inference_zero_shot(
                        text_with_tag,
                        reference_audio,
                        stream=True
                    )
                else:
                    # Cross-lingual mode (uses internal speaker)
                    generator = model.inference_cross_lingual(
                        text_with_tag,
                        stream=True
                    )

                for chunk in generator:
                    if hasattr(chunk, 'numpy'):
                        chunks.append(chunk.numpy())
                    elif hasattr(chunk, 'cpu'):
                        chunks.append(chunk.cpu().numpy())
                    else:
                        chunks.append(chunk)
                    total_samples += len(chunks[-1]) if hasattr(chunks[-1], '__len__') else 0

                if chunks:
                    import numpy as np
                    audio_data = np.concatenate([c.flatten() for c in chunks])
            else:
                # Non-streaming mode
                print("  Generating without streaming...")
                if reference_audio and os.path.exists(reference_audio):
                    result = model.inference_zero_shot(
                        text_with_tag,
                        reference_audio,
                        stream=False
                    )
                else:
                    result = model.inference_cross_lingual(
                        text_with_tag,
                        stream=False
                    )

                if hasattr(result, '__iter__') and not hasattr(result, '__len__'):
                    # Generator result
                    for chunk in result:
                        if hasattr(chunk, 'numpy'):
                            audio_data = chunk.numpy() if audio_data is None else np.concatenate([audio_data, chunk.numpy()])
                        elif hasattr(chunk, 'cpu'):
                            audio_data = chunk.cpu().numpy() if audio_data is None else np.concatenate([audio_data, chunk.cpu().numpy()])
                else:
                    audio_data = result.numpy() if hasattr(result, 'numpy') else result

            gen_time = time.time() - start_time

            if audio_data is not None and len(audio_data) > 0:
                # CosyVoice3 outputs at 22050 Hz
                sample_rate = 22050
                duration = len(audio_data.flatten()) / sample_rate

                print(f"  Generated: {duration:.2f}s audio in {gen_time:.2f}s")
                print(f"  Real-time factor: {gen_time/duration:.2f}x (lower is better)")

                # Save audio
                output_path = output_dir / f"cosyvoice3_test_{i+1}.wav"
                try:
                    import scipy.io.wavfile as wavfile
                    wavfile.write(str(output_path), sample_rate, audio_data.flatten())
                    print(f"  Saved to: {output_path}")
                except ImportError:
                    print("  scipy not available, skipping file save")

                results.append({
                    "text": text,
                    "duration": duration,
                    "gen_time": gen_time,
                    "rtf": gen_time / duration,
                    "success": True
                })
            else:
                print("  No audio generated")
                results.append({"text": text, "success": False})

        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"text": text, "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        print(f"\nAverage real-time factor: {avg_rtf:.2f}x")
        if avg_rtf <= 1.0:
            print("  Real-time synthesis achieved!")
        else:
            print(f"  Synthesis is {avg_rtf:.1f}x slower than real-time")

    print(f"\nOutput files saved to: {output_dir}")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Test CosyVoice3 Korean TTS")
    parser.add_argument(
        "--model",
        type=str,
        default="/path/to/model",
        help="Path to the CosyVoice3 model directory"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Enable streaming mode (default: True)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference audio file for zero-shot voice cloning (3-10 seconds)"
    )
    args = parser.parse_args()

    streaming = args.streaming and not args.no_streaming

    success = test_cosyvoice3_korean(args.model, streaming, args.reference)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
