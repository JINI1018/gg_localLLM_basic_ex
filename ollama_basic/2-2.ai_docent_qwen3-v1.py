# 실습 : image to text
import ollama

response = ollama.chat(
    model="qwen3-vl:8b",
    messages=[
        {
            "role": "user",
            "content": "이 이미지를 한국어로 설명해줘. 주요 객체, 배경, 상황을 구분해서 설명해줘.",
            "images": ["imgs/img01.jpg"]
        }
    ]
)

print(response["message"]["content"])