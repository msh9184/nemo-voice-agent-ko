#!/usr/bin/env python3
"""
OpenAudio S1-mini Korean TTS Test Script (HTTP API Mode)

This script tests OpenAudio S1-mini using the HTTP API server.
This is the RECOMMENDED approach for real-time streaming voice agents.

Benefits over subprocess mode:
- Model loaded once, stays in GPU memory
- ~1-2 second latency instead of 60-360 seconds
- Suitable for real-time streaming

Prerequisites:
1. Start the Fish Speech API server first:
   cd /path/to/workspace
   export PYTHONPATH="/path/to/workspace"
   python -m tools.api_server \
       --listen 0.0.0.0:8080 \
       --llama-checkpoint-path "/path/to/model" \
       --decoder-checkpoint-path "/path/to/model" \
       --decoder-config-name modded_dac_vq
       --compile

2. Then run this test script:
   python test_fish_speech_api.py --server http://localhost:8080

Usage:
    python test_fish_speech_api.py --server http://localhost:8080
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path
from typing import Optional


def test_server_health(server_url: str) -> bool:
    """Check if the Fish Speech API server is running."""
    try:
        # Try common health check endpoints
        for endpoint in ["/", "/health", "/v1/health"]:
            try:
                response = requests.get(f"{server_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"  Server health check passed: {endpoint}")
                    return True
            except:
                continue

        # Try the OpenAPI docs endpoint
        response = requests.get(f"{server_url}/docs", timeout=5)
        if response.status_code == 200:
            print(f"  Server is running (OpenAPI docs available)")
            return True

        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print(f"  Health check error: {e}")
        return False


def generate_audio_api(
    text: str,
    server_url: str,
    output_file: str,
    reference_audio_path: Optional[str] = None,
    reference_audio_text: Optional[str] = None,
) -> bool:
    """
    Generate audio using Fish Speech HTTP API.

    Args:
        text: Korean text to convert to speech
        server_url: Fish Speech API server URL
        output_file: Path to save output audio file
        reference_audio_path: Path to reference audio for voice cloning
        reference_audio_text: Transcript of the reference audio

    Returns:
        True if successful, False otherwise
    """
    try:
        # Prepare request
        # Fish Speech API typically uses /v1/tts or similar endpoint
        endpoints_to_try = [
            "/v1/tts",
            "/api/tts",
            "/tts",
            "/v1/audio/speech",
        ]

        # Build request payload
        payload = {
            "text": text,
        }

        if reference_audio_text:
            payload["reference_text"] = reference_audio_text

        files = None
        if reference_audio_path and os.path.exists(reference_audio_path):
            files = {
                "reference_audio": open(reference_audio_path, "rb")
            }

        # Try different endpoints
        response = None
        for endpoint in endpoints_to_try:
            try:
                url = f"{server_url}{endpoint}"
                print(f"  Trying endpoint: {url}")

                if files:
                    response = requests.post(url, data=payload, files=files, timeout=120)
                else:
                    response = requests.post(url, json=payload, timeout=120)

                if response.status_code == 200:
                    break
                elif response.status_code == 404:
                    continue
                else:
                    print(f"  Endpoint {endpoint} returned {response.status_code}: {response.text[:200]}")

            except requests.exceptions.ConnectionError:
                print(f"  Connection error for {endpoint}")
                continue
            except Exception as e:
                print(f"  Error with {endpoint}: {e}")
                continue

        if files:
            files["reference_audio"].close()

        if response is None:
            print("  ERROR: No endpoint responded successfully")
            return False

        if response.status_code != 200:
            print(f"  ERROR: API returned {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False

        # Save audio response
        content_type = response.headers.get("content-type", "")
        if "audio" in content_type or len(response.content) > 1000:
            with open(output_file, "wb") as f:
                f.write(response.content)
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
        else:
            print(f"  ERROR: Response is not audio. Content-Type: {content_type}")
            print(f"  Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fish_speech_api(
    server_url: str,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    output_dir: Optional[str] = None
) -> bool:
    """
    Test OpenAudio S1-mini Korean TTS using HTTP API.

    Args:
        server_url: Fish Speech API server URL
        reference_audio: Path to Korean reference audio (10-30 seconds)
        reference_text: Transcript of the reference audio
        output_dir: Directory to save output files

    Returns:
        True if all tests passed, False otherwise
    """
    print("\n" + "=" * 80)
    print("OpenAudio S1-mini Korean TTS Test (HTTP API Mode)")
    print("=" * 80)
    print("\nThis is the RECOMMENDED mode for real-time streaming voice agents.")
    print("Model is loaded once and stays in GPU memory for fast inference.")
    print("=" * 80)

    print(f"\nServer URL: {server_url}")

    # Check server health
    print("\n" + "-" * 80)
    print("Checking server health...")
    print("-" * 80)

    if not test_server_health(server_url):
        print(f"\nERROR: Fish Speech API server not responding at {server_url}")
        print("\nPlease start the server first:")
        print("  cd /path/to/workspace")
        print("  export PYTHONPATH=/path/to/workspace")
        print("  python -m tools.api_server \\")
        print("      --listen 0.0.0.0:8080 \\")
        print("      --llama-checkpoint-path /path/to/model \\")
        print("      --decoder-checkpoint-path /path/to/model \\")
        print("      --decoder-config-name modded_dac_vq \\")
        print("      --compile")
        return False

    print("  Server is running!")

    # Check reference audio
    if reference_audio and os.path.exists(reference_audio):
        file_size = os.path.getsize(reference_audio)
        print(f"\nReference audio: {reference_audio}")
        print(f"  File size: {file_size / 1024:.1f} KB")
    elif reference_audio:
        print(f"\nWARNING: Reference audio not found: {reference_audio}")
        reference_audio = None

    if reference_text:
        print(f"Reference text: '{reference_text}'")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "test_outputs" / "openaudio_s1_mini_api"
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
    print("Testing Korean TTS with OpenAudio S1-mini (HTTP API Mode)")
    print("-" * 80)

    results = []
    for i, test in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Text: {test['text']}")

        output_file = str(output_dir / f"{test['name']}.wav")

        start_time = time.time()

        success = generate_audio_api(
            text=test['text'],
            server_url=server_url,
            output_file=output_file,
            reference_audio_path="/path/to/reference.wav",
            reference_audio_text="/path/to/reference.txt",
        )

        gen_time = time.time() - start_time

        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"  SUCCESS: {output_file}")
            print(f"    File size: {file_size / 1024:.1f} KB")
            print(f"    Generation time: {gen_time:.2f}s")

            # Performance assessment
            if gen_time < 2:
                print(f"    Performance: EXCELLENT (suitable for real-time)")
            elif gen_time < 5:
                print(f"    Performance: GOOD (acceptable for real-time)")
            else:
                print(f"    Performance: SLOW (may cause latency issues)")

            results.append({
                "name": test["name"],
                "success": True,
                "output_file": output_file,
                "gen_time": gen_time,
                "file_size": file_size
            })
        else:
            print(f"  FAILED")
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
        avg_time = sum(r["gen_time"] for r in successful) / len(successful)
        print(f"\nAverage generation time: {avg_time:.2f}s")

        if avg_time < 2:
            print("Overall Performance: EXCELLENT - Ready for real-time voice agent")
        elif avg_time < 5:
            print("Overall Performance: GOOD - Acceptable for voice agent")
        else:
            print("Overall Performance: NEEDS IMPROVEMENT")

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
        print("  1. Listen to the generated audio files")
        print("  2. Integrate FishSpeechAPIService into Voice Agent server")
        print("  3. Configure korean_voice_agent_fish_speech.yaml to use API mode")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenAudio S1-mini Korean TTS (HTTP API Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Start server first (in another terminal):
  cd /path/to/workspace
  python -m tools.api_server --listen 0.0.0.0:8080 \\
      --llama-checkpoint-path "/path/to/model" \\
      --decoder-checkpoint-path "/path/to/model"

  # Then run tests:
  python test_fish_speech_api.py --server http://localhost:8080

  # With reference audio for voice cloning:
  python test_fish_speech_api.py --server http://localhost:8080 \\
      --reference /path/to/korean_voice.wav \\
      --reference_text "참조 오디오의 정확한 내용"
        """
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8080",
        help="Fish Speech API server URL"
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

    success = test_fish_speech_api(
        server_url=args.server,
        reference_audio=args.reference,
        reference_text=args.reference_text,
        output_dir=args.output
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
