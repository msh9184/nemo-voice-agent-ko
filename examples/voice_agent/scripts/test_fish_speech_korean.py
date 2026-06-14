#!/usr/bin/env python3
"""
OpenAudio S1-mini Korean TTS Test Script (Subprocess Mode)

This script tests OpenAudio S1-mini for Korean zero-shot TTS using subprocess mode
to avoid dependency conflicts with the existing Korean Voice Agent environment.

Features:
- 0.5B parameters (distilled from 4B S1 model)
- 2M+ hours training data with RLHF
- Zero-shot voice cloning with 10-30s reference audio
- 13 language support including Korean
- 45+ emotion markers support

Usage:
    python test_fish_speech_korean.py \
        --model /path/to/model \
        --fish-speech-repo /path/to/workspace \
        --reference /path/to/korean_reference.wav \
        --reference_text "참조 오디오의 정확한 텍스트"

References:
- GitHub: https://github.com/fishaudio/fish-speech
- HuggingFace: https://huggingface.co/fishaudio/openaudio-s1-mini
- Docs: https://speech.fish.audio/
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def find_fish_speech_repo(custom_path: Optional[str] = None) -> Optional[str]:
    """Find the fish-speech repository."""
    # Check user-specified path first
    if custom_path and os.path.exists(custom_path):
        if os.path.exists(os.path.join(custom_path, "fish_speech")):
            return custom_path
        else:
            print(f"WARNING: {custom_path} exists but fish_speech directory not found inside")

    # Check common locations
    fish_speech_paths = [
        "/path/to/workspace",  # User's preferred location
        "/path/to/workspace",
        os.path.expanduser("~/fish-speech"),
        "/opt/fish-speech",
    ]

    for path in fish_speech_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "fish_speech")):
            return path

    return None


def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment...")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  CUDA: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  WARNING: CUDA not available, will use CPU (slower)")

        return True
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def generate_audio_subprocess(
    text: str,
    model_path: str,
    fish_speech_dir: str,
    output_file: str,
    reference_audio_path: Optional[str] = None,
    reference_audio_text: Optional[str] = None,
) -> bool:
    """
    Generate audio using subprocess calls to OpenAudio/Fish Speech CLI.

    This is the recommended mode to avoid dependency conflicts with the
    existing Korean Voice Agent environment.

    Args:
        text: Korean text to convert to speech
        model_path: Path to OpenAudio S1-mini model directory
        fish_speech_dir: Path to cloned fish-speech repository
        output_file: Path to save output audio file
        reference_audio_path: Path to reference audio for voice cloning
        reference_audio_text: Transcript of the reference audio

    Returns:
        True if successful, False otherwise
    """
    try:
        codec_path = os.path.join(model_path, "codec.pth")

        # Set up environment with PYTHONPATH pointing to fish-speech repo
        # This is required because we don't run 'pip install -e .' to avoid dependency conflicts
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{fish_speech_dir}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = fish_speech_dir

        print(f"  Using PYTHONPATH: {env['PYTHONPATH']}")

        # Create temp directory for intermediate files
        # Fish Speech CLI outputs to fixed filenames in the working directory:
        # - dac/inference.py (encode): outputs "fake.npy"
        # - text2semantic/inference.py: outputs "codes_0.npy"
        # - dac/inference.py (decode): outputs "fake.wav"
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Extract reference tokens (if reference audio provided)
            prompt_args = []
            if reference_audio_path and os.path.exists(reference_audio_path):
                print(f"  Extracting VQ tokens from reference audio...")

                # Extract VQ tokens from reference (outputs fake.npy in cwd)
                result = subprocess.run([
                    sys.executable, f"{fish_speech_dir}/fish_speech/models/dac/inference.py",
                    "-i", reference_audio_path,
                    "--checkpoint-path", codec_path,
                ], capture_output=True, text=True, cwd=tmpdir, env=env)

                if result.returncode != 0:
                    print(f"  ERROR extracting reference tokens: {result.stderr}")
                    if result.stdout:
                        print(f"  STDOUT: {result.stdout}")
                    return False

                ref_tokens_file = os.path.join(tmpdir, "fake.npy")
                if not os.path.exists(ref_tokens_file):
                    print(f"  ERROR: Reference tokens file not created: {ref_tokens_file}")
                    return False

                print(f"  Reference tokens extracted: {ref_tokens_file}")
                prompt_args = [
                    "--prompt-text", reference_audio_text or "",
                    "--prompt-tokens", ref_tokens_file,
                ]

            # Step 2: Generate semantic tokens from text
            # Outputs codes_0.npy in cwd (no -o option available)
            # NOTE: --compile causes slow first-run JIT compilation (~5min)
            # For testing, we skip --compile to see actual behavior faster
            print(f"  Generating semantic tokens from text...")
            print(f"  NOTE: First run may take several minutes due to model loading...")

            cmd = [
                sys.executable, f"{fish_speech_dir}/fish_speech/models/text2semantic/inference.py",
                "--text", text,
                "--checkpoint-path", model_path,
                # "--compile",  # Disabled: causes 5+ min JIT compilation per subprocess
            ] + prompt_args

            print(f"  Command: {' '.join(cmd[:5])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir, env=env)

            # Always print output for debugging
            if result.stdout:
                print(f"  STDOUT: {result.stdout[:1000]}...")  # First 1000 chars
            if result.stderr:
                print(f"  STDERR: {result.stderr[:1000]}...")  # First 1000 chars

            if result.returncode != 0:
                print(f"  ERROR: Process returned code {result.returncode}")
                return False

            # Check all possible output locations
            codes_file = os.path.join(tmpdir, "codes_0.npy")
            codes_file_alt = os.path.join(tmpdir, "temp", "codes_0.npy")

            if os.path.exists(codes_file):
                print(f"  Codes file found: {codes_file}")
            elif os.path.exists(codes_file_alt):
                print(f"  Codes file found in temp subdir: {codes_file_alt}")
                codes_file = codes_file_alt
            else:
                print(f"  ERROR: Codes file not created: {codes_file}")
                # List files in tmpdir for debugging (recursive)
                print(f"  Files in tmpdir:")
                for root, dirs, files in os.walk(tmpdir):
                    level = root.replace(tmpdir, '').count(os.sep)
                    indent = '    ' * level
                    print(f"{indent}{os.path.basename(root)}/")
                    for file in files:
                        print(f"{indent}  {file}")
                return False

            print(f"  Semantic tokens generated: {codes_file}")

            # Step 3: Convert tokens to audio
            # Outputs fake.wav in cwd (no -o option available)
            print(f"  Converting tokens to audio...")
            result = subprocess.run([
                sys.executable, f"{fish_speech_dir}/fish_speech/models/dac/inference.py",
                "-i", codes_file,
                "--checkpoint-path", codec_path,
            ], capture_output=True, text=True, cwd=tmpdir, env=env)

            if result.returncode != 0:
                print(f"  ERROR converting tokens to audio: {result.stderr}")
                if result.stdout:
                    print(f"  STDOUT: {result.stdout}")
                return False

            # Fish Speech outputs to fake.wav in the working directory
            generated_audio = os.path.join(tmpdir, "fake.wav")
            if not os.path.exists(generated_audio):
                print(f"  ERROR: Audio file not created: {generated_audio}")
                print(f"  Files in tmpdir: {os.listdir(tmpdir)}")
                return False

            # Copy to final output location
            import shutil
            shutil.copy2(generated_audio, output_file)
            print(f"  Audio generated and saved to: {output_file}")

            return os.path.exists(output_file)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fish_speech_korean(
    model_path: str,
    fish_speech_repo: Optional[str] = None,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    output_dir: Optional[str] = None
) -> bool:
    """
    Test OpenAudio S1-mini Korean TTS using subprocess mode.

    Args:
        model_path: Path to OpenAudio S1-mini model directory
        fish_speech_repo: Path to cloned fish-speech repository
        reference_audio: Path to Korean reference audio (10-30 seconds)
        reference_text: Transcript of the reference audio
        output_dir: Directory to save output files

    Returns:
        True if all tests passed, False otherwise
    """
    print("\n" + "=" * 80)
    print("OpenAudio S1-mini Korean TTS Test (Subprocess Mode)")
    print("=" * 80)
    print("\nOpenAudio S1-mini Features:")
    print("  - 0.5B parameters (distilled from 4B S1)")
    print("  - 2M+ hours training data with RLHF")
    print("  - Zero-shot voice cloning (10-30s reference audio)")
    print("  - 13 languages: EN, ZH, JA, KO, DE, FR, ES, AR, RU, NL, IT, PL, PT")
    print("  - 45+ emotion markers support")
    print("=" * 80)

    # Validate model path
    if not os.path.exists(model_path):
        print(f"\nERROR: Model path does not exist: {model_path}")
        print("\nTo download the model, run:")
        print(f"  hf download fishaudio/openaudio-s1-mini --local-dir {model_path}")
        return False

    print(f"\nModel path: {model_path}")

    # Check for required model files
    codec_path = os.path.join(model_path, "codec.pth")
    if not os.path.exists(codec_path):
        print(f"ERROR: codec.pth not found in {model_path}")
        print("Make sure the model was downloaded correctly.")
        return False
    print(f"✓ Found codec.pth")

    # Find fish-speech repository
    fish_speech_dir = find_fish_speech_repo(fish_speech_repo)
    if fish_speech_dir is None:
        print(f"\nERROR: Fish Speech repository not found!")
        print("\nPlease clone the repository:")
        print("  git clone https://github.com/fishaudio/fish-speech.git \\")
        print("      /path/to/workspace")
        print("\nOr specify the path with --fish-speech-repo option")
        return False

    print(f"✓ Fish Speech repo found: {fish_speech_dir}")

    # Check environment
    print("\n" + "-" * 80)
    print("Checking environment...")
    print("-" * 80)

    if not check_environment():
        print("\nEnvironment check failed.")
        return False

    # Check reference audio
    if reference_audio and os.path.exists(reference_audio):
        file_size = os.path.getsize(reference_audio)
        estimated_duration = file_size / 32000  # rough estimate
        print(f"\nReference audio: {reference_audio}")
        print(f"  File size: {file_size / 1024:.1f} KB")
        print(f"  Estimated duration: {estimated_duration:.1f}s (10-30s recommended)")
    elif reference_audio:
        print(f"\nWARNING: Reference audio not found: {reference_audio}")
        reference_audio = None

    if reference_text:
        print(f"Reference text: '{reference_text}'")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "test_outputs" / "openaudio_s1_mini"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Test cases
    test_cases = [
        {
            "name": "korean_greeting",
            "text": "안녕하세요",
            "description": "Basic Korean greeting"
        },
        {
            "name": "korean_sentence",
            "text": "반갑습니다. 저는 인공지능 음성 비서입니다.",
            "description": "Korean sentence"
        },
        {
            "name": "korean_question",
            "text": "오늘 날씨가 어떤가요?",
            "description": "Korean question"
        },
        {
            "name": "korean_long",
            "text": "안녕하세요, 오늘 날씨가 정말 좋네요. 어떤 음악을 들려드릴까요?",
            "description": "Longer Korean text"
        }
    ]

    print("\n" + "-" * 80)
    print("Testing Korean TTS with OpenAudio S1-mini (Subprocess Mode)")
    print("-" * 80)

    results = []
    for i, test in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Text: {test['text']}")

        output_file = str(output_dir / f"{test['name']}.wav")

        start_time = time.time()

        success = generate_audio_subprocess(
            text=test['text'],
            model_path=model_path,
            fish_speech_dir=fish_speech_dir,
            output_file=output_file,
            reference_audio_path=reference_audio,
            reference_audio_text=reference_text,
        )

        gen_time = time.time() - start_time

        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"  ✓ SUCCESS: {output_file}")
            print(f"    File size: {file_size / 1024:.1f} KB")
            print(f"    Generation time: {gen_time:.2f}s")
            results.append({
                "name": test["name"],
                "success": True,
                "output_file": output_file,
                "gen_time": gen_time,
                "file_size": file_size
            })
        else:
            print(f"  ✗ FAILED")
            print(f"    Generation time: {gen_time:.2f}s")
            results.append({
                "name": test["name"],
                "success": False,
                "reason": "Audio generation failed"
            })

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
        print(f"\nGenerated audio files:")
        for r in successful:
            print(f"  - {r['output_file']}")
            print(f"    Size: {r['file_size'] / 1024:.1f} KB, Time: {r['gen_time']:.2f}s")

    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('reason', 'Unknown error')}")

    print("\n" + "=" * 80)

    if len(successful) > 0:
        print("\nNext steps:")
        print("  1. Listen to the generated audio files in the output directory")
        print("  2. Test with the Voice Agent server:")
        print("     SERVER_CONFIG_PATH=./server_configs/korean_voice_agent_fish_speech.yaml python server.py")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenAudio S1-mini Korean TTS (Subprocess Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic test (random voice):
  python test_fish_speech_korean.py \\
      --model /path/to/model \\
      --fish-speech-repo /path/to/workspace

  # With reference audio (zero-shot voice cloning):
  python test_fish_speech_korean.py \\
      --model /path/to/model \\
      --fish-speech-repo /path/to/workspace \\
      --reference /path/to/korean_voice.wav \\
      --reference_text "참조 오디오의 정확한 내용"

Prerequisites:
  1. Download model:
     hf download fishaudio/openaudio-s1-mini \\
         --local-dir /path/to/model

  2. Clone Fish Speech repo (do NOT run pip install -e .):
     git clone https://github.com/fishaudio/fish-speech.git \\
         /path/to/workspace
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/path/to/model",
        help="Path to OpenAudio S1-mini model directory"
    )
    parser.add_argument(
        "--fish-speech-repo",
        type=str,
        default="/path/to/workspace",
        help="Path to cloned fish-speech repository"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning (10-30 seconds)"
    )
    parser.add_argument(
        "--reference_text",
        type=str,
        default=None,
        help="Transcript of the reference audio"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for generated audio"
    )
    args = parser.parse_args()

    success = test_fish_speech_korean(
        model_path=args.model,
        fish_speech_repo=args.fish_speech_repo,
        reference_audio=args.reference,
        reference_text=args.reference_text,
        output_dir=args.output
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
