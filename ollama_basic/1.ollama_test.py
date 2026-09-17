import requests

# 로컬에서 실행 중인 Ollama의 텍스트 생성 API 주소
url = "http://localhost:11434/api/generate"

# Ollama API에 전달할 요청 데이터
payload = {
    # 사용할 모델을 지정합니다.
    # 아래 모델 중 실제로 설치된 모델 하나만 활성화해야 합니다.
    # "model": "llama3.2:3b",
    "model": "llama3.2:latest",
    # "model": "exaone3.5:7.8b",

    # 모델에 전달할 질문
    "prompt": "로컬 LLM 기반 앱 개발을 배우는 이유를 3문장으로 설명해줘.",

    # False로 설정하면 응답을 나누어 받지 않고 완성된 결과를 한 번에 받습니다.
    "stream": False,
}

try:
    # JSON 형식으로 요청을 보내고 최대 120초 동안 응답을 기다립니다.
    response = requests.post(url, json=payload, timeout=120)

    # HTTP 상태 코드가 4xx 또는 5xx이면 HTTPError 예외를 발생시킵니다.
    response.raise_for_status()

    # JSON 응답을 파이썬 딕셔너리로 변환합니다.
    data = response.json()

    # Ollama가 생성한 답변을 출력합니다.
    print("모델 응답:")
    print(data["response"])
except requests.exceptions.ConnectionError:
    # Ollama 서버가 실행 중이지 않거나 주소가 잘못된 경우
    print("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
except requests.exceptions.Timeout:
    # 지정한 120초 안에 응답을 받지 못한 경우
    print("요청 시간이 초과되었습니다. 모델 로딩 또는 PC 성능 문제일 수 있습니다.")
except requests.exceptions.HTTPError as e:
    # 모델이 설치되지 않았거나 API 요청이 잘못된 경우
    print(f"HTTP 오류가 발생했습니다: {e}")
except Exception as e:
    # 위에서 처리하지 못한 기타 오류
    print(f"알 수 없는 오류가 발생했습니다: {e}")
