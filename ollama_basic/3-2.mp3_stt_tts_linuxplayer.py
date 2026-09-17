###############################################
# STT → Ollama → TTS
# 음성 파일입력(mp3) → 텍스트  
# →  Ollama: 텍스트 질문 → 텍스트 답변  
# → TTS: 텍스트 답변 → 스피커 음성 출력

# STT : faster-whisper
# TTS : edge-tts
# uv pip install ollama faster-whisper edge-tts pygame

# sudo apt update
# sudo apt install -y libpulse0 pulseaudio-utils libasound2-plugins

# ./voice/voice1.mp3
# → faster-whisper STT
# → Ollama 답변 생성
# → edge-tts TTS
# → ./voice/answer.mp3 저장
# → pygame으로 음성 출력
###############################################
import asyncio
import os
import platform
import time
from pathlib import Path

# pygame 안내 문구 숨기기
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def is_wsl() -> bool:
    """현재 실행 환경이 WSL인지 확인한다."""
    try:
        release = platform.uname().release.lower()
        return "microsoft" in release or "wsl" in release
    except Exception:
        return False


# WSL2에서 pygame을 사용할 경우 ALSA 대신 PulseAudio를 우선 시도한다.
# Windows PowerShell 실행 시에는 영향이 거의 없다.
if is_wsl():
    os.environ.setdefault("SDL_AUDIODRIVER", "pulse")

import ollama
from faster_whisper import WhisperModel
import edge_tts
import pygame


# =========================
# 기본 설정
# =========================

# Ollama에 미리 내려받은 모델 이름이다.
# 처음 테스트가 무거우면 llama3.2:3b로 바꾸는 것을 권장한다.
OLLAMA_MODEL = "exaone3.5:7.8b"
# OLLAMA_MODEL = "llama3.2:3b"

# STT 입력 파일과 TTS 출력 파일 경로
AUDIO_FILE = Path("./voice/voice1.mp3")
OUTPUT_TTS_FILE = Path("./voice/answer.mp3")

# Whisper 모델 크기와 연산 장치 설정
WHISPER_MODEL_SIZE = "base"    # tiny, base, small, medium, large-v3
WHISPER_DEVICE = "cpu"         # RTX 3080 + CUDA 환경이면 "cuda"
WHISPER_COMPUTE_TYPE = "int8"  # cuda 사용 시 "float16" 권장

# edge-tts에서 사용할 한국어 음성
TTS_VOICE = "ko-KR-SunHiNeural"

# False로 바꾸면 MP3 생성까지만 수행하고 자동 재생하지 않는다.
AUTO_PLAY = True               # WSL2에서 재생 오류가 나면 False로 변경


# =========================
# STT 모델 로드
# =========================

print("STT 모델 로딩 중...")

# 한 번 로드한 Whisper 모델을 모든 STT 요청에서 재사용한다.
stt_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

print("STT 모델 로딩 완료")


# =========================
# 대화 히스토리: 모델의 역할과 말투를 system 메시지로 먼저 지정한다.
# =========================

messages = [
    {
        "role": "system",
        "content": (
            "너는 한국어로 간결하고 명확하게 답변하는 로컬 AI 비서다. "
            "사용자의 음성 질문을 텍스트로 변환한 내용을 바탕으로 자연스럽게 답변하라."
        )
    }
]


def transcribe_audio(audio_path: Path) -> str:
    """음성 파일을 텍스트로 변환한다."""

    if not audio_path.exists():
        raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

    print(f"\n음성 파일 읽는 중: {audio_path}")
    print("STT 변환 중...")

    # 한국어 음성을 구간별로 인식한다.
    segments, info = stt_model.transcribe(
        str(audio_path),
        language="ko",
        beam_size=5
    )

    # 구간별 결과를 하나의 문자열로 합친다.
    text = " ".join(segment.text.strip() for segment in segments).strip()

    return text


def ask_ollama(user_text: str) -> str:
    """Ollama 모델에 질문하고 답변을 받는다."""

    # 새 질문을 기존 대화 뒤에 추가한다.
    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )

    print("\nOllama 답변 생성 중...")

    # 전체 히스토리를 전달해 문맥을 유지한 답변을 생성한다.
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            # 생성 다양성, 누적 확률 범위, 최대 생성 길이 설정
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 512
        }
    )

    # ollama-python 버전에 따라 객체/딕셔너리 접근 모두 대비
    try:
        answer = response.message.content
    except AttributeError:
        answer = response["message"]["content"]

    answer = answer.strip()

    # 생성된 답변도 다음 대화에서 사용할 수 있도록 저장한다.
    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer


async def text_to_speech(text: str, output_path: Path) -> None:
    """텍스트 답변을 음성 MP3 파일로 변환한다."""

    if not text.strip():
        raise ValueError("TTS로 변환할 텍스트가 비어 있습니다.")

    print("\nTTS 변환 중...")

    # 출력 디렉터리가 없으면 자동 생성한다.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 네트워크 기반 edge-tts로 텍스트를 MP3 음성으로 합성한다.
    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate="+0%",
        volume="+0%"
    )

    await communicate.save(str(output_path))

    print(f"TTS 파일 저장 완료: {output_path}")


def play_audio(audio_path: Path) -> None:
    """pygame을 사용해 MP3 파일을 재생한다."""

    if not audio_path.exists():
        raise FileNotFoundError(f"재생할 음성 파일을 찾을 수 없습니다: {audio_path}")

    print("\n음성 출력 중...")

    # 오디오 믹서를 초기화하고 MP3 파일을 메모리에 불러온다.
    pygame.mixer.init()
    pygame.mixer.music.load(str(audio_path))
    pygame.mixer.music.play()

    # 재생이 끝날 때까지 프로그램이 종료되지 않도록 대기한다.
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()

    print("음성 출력 완료")


def main() -> None:
    """파일 기반 STT → Ollama → TTS 전체 실행 함수."""

    print("\n파일 기반 음성 Ollama 앱 시작")

    print(f"실행 환경: {platform.system()}")
    print(f"WSL 여부: {is_wsl()}")
    print(f"입력 음성 파일: {AUDIO_FILE}")
    print(f"출력 음성 파일: {OUTPUT_TTS_FILE}")

    try:
        # 1. 음성 파일 → 텍스트
        user_text = transcribe_audio(AUDIO_FILE)

        if not user_text:
            print("\n음성을 인식하지 못했습니다.")
            return

        print("\n사용자 음성 인식 결과:")
        print(user_text)

        # 2. 텍스트 → Ollama 답변
        answer = ask_ollama(user_text)

        if not answer:
            print("\nOllama 답변이 비어 있습니다.")
            return

        print("\nAI 답변:")
        print(answer)

        # 3. 답변 텍스트 → 음성 파일
        asyncio.run(text_to_speech(answer, OUTPUT_TTS_FILE))

        # 4. 음성 출력
        if AUTO_PLAY:
            try:
                play_audio(OUTPUT_TTS_FILE)
            except Exception as play_error:
                print("\n음성 파일 생성은 성공했지만, 자동 재생에 실패했습니다.")
                print(f"재생 오류: {type(play_error).__name__}")
                print(play_error)
                print(f"\n생성된 파일을 직접 열어 재생하세요: {OUTPUT_TTS_FILE}")
        else:
            print(f"\n음성 답변 파일이 생성되었습니다: {OUTPUT_TTS_FILE}")
            print("AUTO_PLAY=False 상태이므로 자동 재생은 생략했습니다.")

    except FileNotFoundError as e:
        print("\n파일 오류가 발생했습니다.")
        print(e)

    except ollama.ResponseError as e:
        print("\nOllama 응답 오류가 발생했습니다.")
        print(e)
        print("\n모델이 설치되어 있는지 확인하세요.")
        print(f"예: ollama pull {OLLAMA_MODEL}")

    except ConnectionError as e:
        print("\nOllama 서버 연결 오류가 발생했습니다.")
        print(e)
        print("\nOllama가 실행 중인지 확인하세요.")
        print("예: ollama serve")

    except Exception as e:
        print("\n오류 발생:")
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    # 모듈로 import될 때는 실행하지 않고, 직접 실행할 때만 시작한다.
    main()
