###############################################
# STT → Ollama → TTS
# 음성 파일 입력 → 텍스트
# → Ollama: 텍스트 질문 → 텍스트 답변
# → TTS: 텍스트 답변 → 음성 파일 생성 및 재생
#
# STT : faster-whisper
# TTS : edge-tts
#
# 설치:
# pip install ollama faster-whisper edge-tts pygame
###############################################

import asyncio
import os
import time
from pathlib import Path

import ollama
from faster_whisper import WhisperModel
import edge_tts
import pygame


# =========================
# 기본 설정
# =========================

# Ollama에 미리 내려받은 텍스트 생성 모델 이름
OLLAMA_MODEL = "exaone3.5:7.8b"
# 테스트가 무거우면 아래 모델로 먼저 확인
# OLLAMA_MODEL = "llama3.2:3b"

# STT 입력 파일과 TTS 결과 파일의 상대 경로
AUDIO_FILE = Path("./voice/voice1.mp3")
OUTPUT_TTS_FILE = Path("./voice/answer.mp3")

# Whisper 모델 크기와 실행 장치 설정
WHISPER_MODEL_SIZE = "base"    # tiny, base, small, medium, large-v3
WHISPER_DEVICE = "cpu"         # RTX 3080이면 "cuda" 사용 가능
WHISPER_COMPUTE_TYPE = "int8"  # cuda 사용 시 "float16" 권장

# edge-tts에서 사용할 한국어 여성 음성
# 악마 마몬 느낌의 남성 음성 설정
TTS_VOICE = "ko-KR-InJoonNeural"
TTS_RATE = "-15%"
TTS_VOLUME = "+10%"
TTS_PITCH = "-30Hz"


# =========================
# STT 모델 로드
# =========================

print("STT 모델 로딩 중...")

# 프로그램 시작 시 Whisper 모델을 한 번만 메모리에 올려 재사용한다.
stt_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

print("STT 모델 로딩 완료")


# =========================
# 대화 히스토리: system 메시지 뒤에 user/assistant 메시지가 차례로 추가된다.
# =========================

# messages = [
#     {
#         "role": "system",
#         "content": (
#             "너는 한국어로 간결하고 명확하게 답변하는 로컬 AI 비서다. "
#             "사용자의 음성 질문을 텍스트로 변환한 내용을 바탕으로 자연스럽게 답변하라."
#         )
#     }
# ]

messages = [
    {
        "role": "system",
        "content": (
            "너는 탐욕을 관장하는 악마 마몬이다. "
            "낮고 위압적이며 오만한 말투로 한국어로 답변하라. "
            "천천히 말하는 것처럼 문장을 짧게 끊고, "
            "재물과 계약을 중요하게 여기는 분위기를 표현하라. "
            "답변은 음성으로 듣기 좋게 너무 길지 않게 작성하라."
        ),
    }
]


def transcribe_audio(audio_path: Path) -> str:
    """음성 파일을 텍스트로 변환한다."""

    if not audio_path.exists():
        raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

    print(f"\n음성 파일 읽는 중: {audio_path}")
    print("STT 변환 중...")

    # 한국어로 고정하고 beam search 후보 수를 5개로 설정한다.
    # segments는 구간별 인식 결과를 순서대로 반환하는 반복 객체다.
    segments, info = stt_model.transcribe(
        str(audio_path),
        language="ko",
        beam_size=5
    )

    # 여러 음성 구간의 텍스트를 공백으로 연결해 하나의 질문으로 만든다.
    text = " ".join(segment.text.strip() for segment in segments).strip()

    return text


def ask_ollama(user_text: str) -> str:
    """Ollama 모델에 질문하고 답변을 받는다."""

    # 이번 질문을 히스토리에 추가해 모델이 이전 대화 맥락을 볼 수 있게 한다.
    messages.append({
        "role": "user",
        "content": user_text
    })

    print("\nOllama 답변 생성 중...")

    # stream을 지정하지 않았으므로 완성된 응답 객체를 한 번에 받는다.
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            # 낮은 temperature는 답변의 무작위성을 줄인다.
            "temperature": 0.3,
            # 확률 누적 상위 90% 안에서 다음 토큰을 선택한다.
            "top_p": 0.9,
            # 생성할 최대 토큰 수를 제한한다.
            "num_predict": 512
        }
    )

    # ollama-python 버전에 따라 객체/딕셔너리 접근 모두 대비
    try:
        answer = response.message.content
    except AttributeError:
        answer = response["message"]["content"]

    # 답변 앞뒤의 불필요한 공백과 줄바꿈을 제거한다.
    answer = answer.strip()

    # 모델 답변도 히스토리에 보관해 다음 질문에서 맥락으로 사용한다.
    messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


async def text_to_speech(text: str, output_path: Path) -> None:
    """텍스트 답변을 음성 MP3 파일로 변환한다."""

    print("\nTTS 변환 중...")

    # voice 폴더가 없으면 상위 폴더까지 함께 생성한다.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # edge-tts는 비동기 API이므로 communicate.save()를 await로 기다린다.
    communicate = edge_tts.Communicate(
        text=text,
        voice=TTS_VOICE,
        rate=TTS_RATE,
        volume=TTS_VOLUME,
        pitch=TTS_PITCH
    )

    await communicate.save(str(output_path))

    print(f"TTS 파일 저장 완료: {output_path}")


def play_audio(audio_path: Path) -> None:
    """WSL2에서는 Windows 기본 플레이어로 MP3 파일을 연다."""

    if not audio_path.exists():
        raise FileNotFoundError(f"재생할 음성 파일을 찾을 수 없습니다: {audio_path}")

    print("\n음성 출력 중...")

    try:
        # WSL2 환경이면 Windows 경로로 변환 후 Windows 기본 플레이어 실행
        import subprocess

        # WSL의 Linux 경로를 Windows 프로그램이 이해하는 경로로 변환한다.
        linux_path = str(audio_path.resolve())
        windows_path = subprocess.check_output(
            ["wslpath", "-w", linux_path],
            text=True
        ).strip()

        # PowerShell의 Start-Process로 Windows 기본 MP3 플레이어를 연다.
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"Start-Process -FilePath '{windows_path}'"
            ],
            check=True
        )

        print(f"Windows 기본 플레이어로 재생 파일을 열었습니다: {windows_path}")

    except Exception as e:
        print("Windows 플레이어 실행에 실패했습니다.")
        print(e)

def main():
    """파일 입력부터 Windows 재생까지 전체 파이프라인을 실행한다."""
    print("\n파일 기반 음성 Ollama 앱 시작")

    try:
        # 1. 음성 파일 → 텍스트
        user_text = transcribe_audio(AUDIO_FILE)

        if not user_text:
            print("음성을 인식하지 못했습니다.")
            return

        print("\n사용자 음성 인식 결과:")
        print(user_text)

        # 2. 텍스트 → Ollama 답변
        answer = ask_ollama(user_text)

        print("\nAI 답변:")
        print(answer)

        # 3. 답변 텍스트 → 음성 파일
        asyncio.run(text_to_speech(answer, OUTPUT_TTS_FILE))

        # 4. 음성 출력
        play_audio(OUTPUT_TTS_FILE)

    except Exception as e:
        print(f"\n오류 발생: {type(e).__name__}")
        print(e)


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 main()을 호출한다.
    main()
