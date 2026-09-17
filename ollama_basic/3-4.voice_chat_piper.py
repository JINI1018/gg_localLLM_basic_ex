# 미션 : 문제 해결
###############################################
# WSL Linux STT → Ollama → TTS 음성 챗 앱
#
# 마이크 음성 입력
# → STT: faster-whisper
# → LLM: Ollama gemma4:e4b
# → TTS: edge-tts
# → 출력: mpg123 또는 pygame
#
# 설치:
# sudo apt install -y libportaudio2 portaudio19-dev libasound2-dev \
#   libpulse0 pulseaudio-utils libasound2-plugins alsa-utils
#
# mkdir -p models/piper

# wget -O models/piper/piper-kss-korean.onnx \
# https://huggingface.co/neurlang/piper-onnx-kss-korean/resolve/main/piper-kss-korean.onnx

# wget -O models/piper/piper-kss-korean.onnx.json \
# https://huggingface.co/neurlang/piper-onnx-kss-korean/resolve/main/piper-kss-korean.onnx.json
###############################################
###############################################
# WSL Linux STT → Ollama → Piper TTS 음성 챗 앱
#
# 마이크 음성 입력
# → STT: faster-whisper
# → LLM: Ollama / gemma4:e4b
# → TTS: Piper
# → 출력: paplay / aplay / pygame
#
# 설치:
# sudo apt install -y libportaudio2 portaudio19-dev libasound2-dev \
#   libpulse0 pulseaudio-utils libasound2-plugins alsa-utils
# uv pip install ollama sounddevice scipy faster-whisper pygame piper-tts

# WSL에서 마이크 인식시키기
# pactl list short sources
# pactl info | grep "Default Source"
# pactl set-default-source RDPSource
###############################################

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

# pygame import 전에 안내 문구 출력 여부를 환경 변수로 설정한다.
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

def is_wsl() -> bool:
    """현재 실행 환경이 WSL인지 확인한다."""
    try:
        release = platform.uname().release.lower()
        return "microsoft" in release or "wsl" in release
    except Exception:
        return False


# WSL에서는 pygame이 ALSA 대신 PulseAudio를 우선 사용하도록 설정한다.
# 반드시 pygame import 전에 설정해야 한다.
if is_wsl():
    os.environ.setdefault("SDL_AUDIODRIVER", "pulse")


import ollama
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import pygame


# =========================
# 기본 설정
# =========================

# Ollama에 설치된 모델 중 하나만 활성화한다.
OLLAMA_MODEL = "gemma4:e4b"
# OLLAMA_MODEL = "llama3.2:3b"
# OLLAMA_MODEL = "llama3.2:1b"

# 녹음 샘플레이트와 한 번에 녹음할 시간
SAMPLE_RATE = 44100
RECORD_SECONDS = 5

# faster-whisper 모델 크기와 실행 장치 설정
WHISPER_MODEL_SIZE = "base"      # tiny, base, small, medium, large-v3
WHISPER_DEVICE = "cpu"           # CUDA 가능 시 "cuda"
WHISPER_COMPUTE_TYPE = "int8"    # CUDA 사용 시 "float16"

# 입력 녹음 파일과 Piper 출력 파일 경로
VOICE_DIR = Path("./voice")
INPUT_WAV_FILE = VOICE_DIR / "input.wav"
OUTPUT_WAV_FILE = VOICE_DIR / "answer.wav"

# Piper 음성 모델과 그 모델의 설정 파일
PIPER_MODEL_FILE = Path("./models/piper/piper-kss-korean.onnx")
PIPER_CONFIG_FILE = Path("./models/piper/piper-kss-korean.onnx.json")

# 특정 마이크 장치를 직접 지정해야 하면 숫자로 설정한다.
# None이면 시스템 기본 입력 장치를 사용한다.
INPUT_DEVICE_INDEX = None

# system 메시지를 제외하고 유지할 최근 user/assistant 메시지 수
MAX_HISTORY_MESSAGES = 10


# =========================
# STT 모델 로드
# =========================

print("STT 모델 로딩 중...")

# 모델 로딩 비용이 크므로 앱 시작 시 한 번만 생성하고 재사용한다.
stt_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

print("STT 모델 로딩 완료")


# =========================
# 대화 히스토리: system 메시지로 답변 언어와 길이를 지정한다.
# =========================

messages = [
    {
        "role": "system",
        "content": (
            "너는 한국어로 간결하고 명확하게 답변하는 로컬 AI 비서다. "
            "사용자의 음성 질문에 자연스럽게 답변하라. "
            "답변은 음성으로 들었을 때 이해하기 쉽도록 너무 길지 않게 작성하라."
        )
    }
]


def check_piper_files() -> None:
    """Piper 모델 파일 존재 여부를 확인한다."""

    # Piper 실행 파일이 현재 PATH에 등록되어 있는지 확인한다.
    if shutil.which("piper") is None:
        raise RuntimeError(
            "piper 명령을 찾을 수 없습니다. "
            "pip install piper-tts 설치 후 다시 실행하세요."
        )

    # 음성 합성에는 ONNX 모델과 JSON 설정 파일이 모두 필요하다.
    if not PIPER_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Piper 모델 파일을 찾을 수 없습니다: {PIPER_MODEL_FILE}"
        )

    if not PIPER_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Piper 설정 파일을 찾을 수 없습니다: {PIPER_CONFIG_FILE}"
        )


def show_audio_devices() -> None:
    """현재 WSL/Linux에서 인식되는 오디오 장치를 출력한다."""

    print("\n[오디오 장치 목록]")
    # sounddevice가 PortAudio를 통해 발견한 입출력 장치를 모두 보여준다.
    print(sd.query_devices())
    print(f"\n기본 장치: {sd.default.device}")


def record_audio(output_path: Path, seconds: int = RECORD_SECONDS) -> None:
    """마이크 음성을 WAV 파일로 저장한다."""

    # voice 디렉터리가 없으면 생성한다.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{seconds}초 동안 말하세요.")

    try:
        # 실제 녹음 전에 선택한 장치가 설정을 지원하는지 검사한다.
        sd.check_input_settings(
            device=INPUT_DEVICE_INDEX,
            samplerate=SAMPLE_RATE,
            channels=1
        )
    except Exception as e:
        print("\n입력 장치 설정 확인 중 오류가 발생했습니다.")
        print(f"{type(e).__name__}: {e}")
        print("장치 목록을 확인한 뒤 INPUT_DEVICE_INDEX를 지정해야 할 수 있습니다.")
        raise

    # 녹음할 전체 샘플 수는 녹음 시간 × 초당 샘플 수다.
    # 단일 채널 float32 파형을 비동기로 녹음하기 시작한다.
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE_INDEX
    )

    # 지정한 시간의 녹음이 끝날 때까지 기다린다.
    sd.wait()

    # SciPy를 이용해 메모리의 파형을 WAV 파일로 저장한다.
    write(output_path, SAMPLE_RATE, audio)

    print(f"녹음 완료: {output_path}")


def transcribe_audio(audio_path: Path) -> str:
    """WAV 음성 파일을 텍스트로 변환한다."""

    if not audio_path.exists():
        raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

    print("\nSTT 변환 중...")

    # 한국어로 언어를 고정하고 beam search 후보를 5개 사용한다.
    segments, info = stt_model.transcribe(
        str(audio_path),
        language="ko",
        beam_size=5
    )

    # Whisper가 나눈 구간별 문장을 하나의 질문으로 연결한다.
    text = " ".join(segment.text.strip() for segment in segments).strip()

    return text


def trim_messages() -> None:
    """대화 히스토리가 너무 길어지지 않도록 최근 메시지만 유지한다."""

    global messages

    # system 메시지는 유지하고 오래된 일반 대화만 제거한다.
    system_message = messages[0]
    recent_messages = messages[1:][-MAX_HISTORY_MESSAGES:]

    messages = [system_message] + recent_messages


def ask_ollama(user_text: str) -> str:
    """Ollama 모델에 질문하고 답변을 받는다."""

    # STT로 인식한 질문을 대화 히스토리에 추가한다.
    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )

    # 컨텍스트가 무한히 늘어나지 않도록 최근 대화만 유지한다.
    trim_messages()

    print("\nOllama 답변 생성 중...")

    # stream 기본값은 False이므로 완성된 응답 객체를 반환한다.
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            # 생성 다양성, 확률 후보 범위, 최대 답변 길이 설정
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 512
        }
    )

    # ollama-python 버전에 따른 객체/딕셔너리 형식을 모두 처리한다.
    try:
        answer = response.message.content
    except AttributeError:
        answer = response["message"]["content"]

    answer = answer.strip()

    # 모델 답변도 다음 질문에서 문맥으로 사용할 수 있도록 저장한다.
    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer


def text_to_speech_with_piper(text: str, output_path: Path) -> None:
    """Piper CLI를 사용해 텍스트 답변을 WAV 음성 파일로 변환한다."""

    if not text.strip():
        raise ValueError("Piper로 변환할 텍스트가 비어 있습니다.")

    check_piper_files()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nPiper TTS 변환 중...")

    # Piper CLI에 모델과 출력 WAV 경로를 인자로 전달한다.
    command = [
        "piper",
        "--model",
        str(PIPER_MODEL_FILE),
        "--output_file",
        str(output_path)
    ]

    # 합성할 텍스트는 Piper 프로세스의 표준 입력으로 전달한다.
    result = subprocess.run(
        command,
        input=text,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print("\nPiper 실행 실패")
        print("[stdout]")
        print(result.stdout)
        print("[stderr]")
        print(result.stderr)
        raise RuntimeError("Piper TTS 변환에 실패했습니다.")

    print(f"Piper TTS 파일 저장 완료: {output_path}")


def play_with_paplay(audio_path: Path) -> bool:
    """PulseAudio paplay로 WAV 파일을 재생한다."""

    # 명령이 설치되어 있지 않으면 다음 재생 방법을 시도하게 한다.
    if shutil.which("paplay") is None:
        return False

    try:
        subprocess.run(["paplay", str(audio_path)], check=True)
        return True
    except Exception as e:
        print("\npaplay 재생 실패")
        print(f"{type(e).__name__}: {e}")
        return False


def play_with_aplay(audio_path: Path) -> bool:
    """ALSA aplay로 WAV 파일을 재생한다."""

    # ALSA 재생기 설치 여부를 먼저 확인한다.
    if shutil.which("aplay") is None:
        return False

    try:
        subprocess.run(["aplay", str(audio_path)], check=True)
        return True
    except Exception as e:
        print("\naplay 재생 실패")
        print(f"{type(e).__name__}: {e}")
        return False


def play_with_pygame(audio_path: Path) -> bool:
    """pygame으로 WAV 파일을 재생한다."""

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.play()

        # WAV 재생이 끝날 때까지 프로그램을 유지한다.
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.quit()
        return True

    except Exception as e:
        print("\npygame 재생 실패")
        print(f"{type(e).__name__}: {e}")

        try:
            pygame.mixer.quit()
        except Exception:
            pass

        return False


def play_audio(audio_path: Path) -> None:
    """WAV 파일을 재생한다."""

    if not audio_path.exists():
        raise FileNotFoundError(f"재생할 음성 파일을 찾을 수 없습니다: {audio_path}")

    print("\n음성 출력 중...")

    # WSL에서는 PulseAudio 기반 paplay가 가장 먼저 시도할 만하다.
    if play_with_paplay(audio_path):
        print("음성 출력 완료")
        return

    # 일반 Linux/ALSA 환경에서는 aplay가 동작할 수 있다.
    if play_with_aplay(audio_path):
        print("음성 출력 완료")
        return

    # 마지막 fallback
    if play_with_pygame(audio_path):
        print("음성 출력 완료")
        return

    print("\n자동 재생에 실패했습니다.")
    print(f"생성된 파일을 직접 열어 재생하세요: {audio_path}")


def run_voice_turn() -> None:
    """음성 입력 1회에 대한 STT → Ollama → Piper TTS → 출력 흐름을 실행한다."""

    # 1. 마이크 녹음
    record_audio(INPUT_WAV_FILE)

    # 2. STT
    user_text = transcribe_audio(INPUT_WAV_FILE)

    if not user_text:
        print("\n음성을 인식하지 못했습니다.")
        return

    print("\n사용자 음성 인식 결과:")
    print(user_text)

    # 3. Ollama 답변
    answer = ask_ollama(user_text)

    if not answer:
        print("\nOllama 답변이 비어 있습니다.")
        return

    print("\nAI 답변:")
    print(answer)

    # 4. Piper TTS
    text_to_speech_with_piper(answer, OUTPUT_WAV_FILE)

    # 5. 음성 출력
    play_audio(OUTPUT_WAV_FILE)


def main() -> None:
    """음성 챗 앱 메인 루프."""

    print("\nWSL Linux 음성 Ollama + Piper 챗 앱 시작")
    print(f"실행 환경: {platform.system()}")
    print(f"WSL 여부: {is_wsl()}")
    print(f"Ollama 모델: {OLLAMA_MODEL}")
    print(f"Whisper 모델: {WHISPER_MODEL_SIZE}")
    print(f"Piper 모델: {PIPER_MODEL_FILE}")
    print(f"녹음 시간: {RECORD_SECONDS}초")

    # 대화 루프에 들어가기 전에 Piper 설치와 모델 파일을 검증한다.
    try:
        check_piper_files()
    except Exception as e:
        print("\nPiper 설정 오류")
        print(f"{type(e).__name__}: {e}")
        return

    show_audio_devices()

    print("\n명령:")
    print("Enter 또는 r : 음성 녹음 시작")
    print("d          : 오디오 장치 목록 다시 보기")
    print("q          : 종료")

    # q를 입력하기 전까지 여러 번 음성 질문을 받을 수 있다.
    while True:
        command = input("\n명령 입력: ").strip().lower()

        if command == "q":
            print("종료합니다.")
            break

        if command == "d":
            show_audio_devices()
            continue

        if command not in ["", "r"]:
            print("Enter 또는 r을 입력하면 녹음을 시작합니다. q는 종료입니다.")
            continue

        try:
            run_voice_turn()

        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")
            break

        except ollama.ResponseError as e:
            print("\nOllama 응답 오류가 발생했습니다.")
            print(e)
            print(f"\n모델 설치 여부를 확인하세요: ollama pull {OLLAMA_MODEL}")

        except ConnectionError as e:
            print("\nOllama 서버 연결 오류가 발생했습니다.")
            print(e)
            print("Ollama가 실행 중인지 확인하세요: ollama serve")

        except Exception as e:
            print("\n오류 발생")
            print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    # 다른 모듈에서 import할 때는 실행하지 않고 직접 실행할 때만 시작한다.
    main()
