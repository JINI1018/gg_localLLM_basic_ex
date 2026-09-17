# 실습: 이미지를 분석하여 텍스트로 설명하기(Image to Text)

# Ollama Python 라이브러리에서 대화형 API 함수를 가져옵니다.
from ollama import chat

# 분석할 이미지 파일의 경로
IMAGE_PATH = "imgs/img01.jpg"

# 이미지를 인식할 수 있는 멀티모달 Ollama 모델 이름
# 실행 전에 `ollama list` 명령으로 해당 모델의 설치 여부를 확인합니다.
MODEL_NAME = "gemma4:e4b"

# 모델에 이미지와 질문을 전달하고 응답을 받습니다.
response = chat(
    # 이미지 분석에 사용할 모델
    model=MODEL_NAME,

    # 모델에 전달할 대화 메시지 목록
    messages=[
        {
            # 사용자가 보낸 메시지임을 나타냅니다.
            "role": "user",

            # 모델에 요청할 이미지 설명 형식
            "content": """
이 이미지를 한국어로 설명해줘.

다음 형식으로 답변해줘.
1. 전체 장면
2. 주요 객체
3. 배경
4. 이미지에서 추론 가능한 상황
""",

            # 함께 분석할 이미지 파일을 첨부합니다.
            "images": [IMAGE_PATH],
        }
    ],

    # 모델의 추론 과정은 출력하지 않습니다.
    think=False,

    # 스트리밍하지 않고 완성된 응답을 한 번에 받습니다.
    # stream=False,
    stream=True,
)

# 응답 객체에서 모델이 생성한 텍스트만 꺼내 출력합니다.
# print(response.message.content) # stream=False 설정과 세트
for chunk in response: # stream=True 설정과 세트
    print(chunk.message.content, end="", flush=True)