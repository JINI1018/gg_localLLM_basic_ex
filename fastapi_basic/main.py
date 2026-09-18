from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn # fastapi 내장 웹서버


# FastAPI 객체 생상(필수)
# 라우터 등록 및 엔드포인트 정의는 이 객체를 통해 이루어짐
app = FastAPI()

# DTO: 데이터 전송 객체(Data Transfer Object)
class UserCreate(BaseModel):
    username: str
    password: str
    avartar_url: Optional[HttpUrl] = None
    user_fullname: Optional[str] = None

# DTO: 응답 전송 객체
class UserResponse(BaseModel):
    username: str
    avartar_url: HttpUrl
    user_fullname: Optional[str] = None

# http://localhost:8000/ = 루트 주소('/')에 GET 요청이 오면 read_root 함수 실행
# http://127.0.0.1:8000/ = 루트 주소('/')에 GET 요청이 오면 read_root 함수 실행(동일)
# app은 FastAPI 앱 객체
# .get("/")는 “루트 주소('/')에 GET 요청이 오면”
#그 아래 함수 read_root를 실행해라
# 이와 같은 방식으로 다른 엔드포인트도 정의할 수 있음
# 즉, @는 함수에 추가 정보를 붙이는 문법이고, 여기서는 FastAPI 라우트 등록용으로 쓰이는 것.
@app.get("/")
async def read_root():
    # 비즈니스 로직
    data = "DB에서 데이터 읽어오기"
    return {"message":data} # 반드시 딕셔너리

@app.post("/User_info/", response_model=UserResponse)
def create_user(user: UserCreate):
    # 비즈니스 로직 처리
    print(f"username: {user.username}")
    print(f"avartar_url: {user.avartar_url}")
    print(f"user_fullname: {user.user_fullname}")

    user_info = UserResponse(
        name=user.username,
        avartar_url=user.avartar_url,
    )
    return user_info
    # return user_info{user}

@app.get("/items")
def read_item():
    item_id = 1
    q = "사과"
    return {"item_id": item_id, "q": q}

# http://localhost:8000/items/300?q=치킨 = 특정 아이템 조회
# http://localhost:8000/items/100?q=사과
# 클라이언트에서 변하는 값을 서버로 받을 수 있음
@app.get("/items/{item_id}") # 클라이언트에서 특정 아이템 조회 요청이 오면 read_item 함수 실행
def read_item(item_id: int, q: str | None = None): # item_id는 경로 매개변수, q는 쿼리 매개변수
    # 비즈니스 로직 처리
    print(f"item_id: {item_id}, q: {q}")

    return {"item_id": item_id, "q": q}
# 127.0.0.1:54694 - "GET /items/100?q=%EA%B7%A4 HTTP/1.1" 200 -> 터미널에서 입력값을 받을 때 확인 가능


@app.post("/User_info/{user_id}")
def create_user_info(user_id: int, q: str | None = None):
    # 비즈니스 로직 처리
    print(f"user_id: {user_id}, q: {q}")
    return {"user_id": user_id, "q": q}

# uv run fastapi dev
if __name__ == "__main__":
    # uvicorn.run("main:app", reload=True)
    # # main은 현재_파일 이름: FastAPI 객체_식별자.
    # reload=True: 개발자 모드 실행. 코드 변경 시 서버를 자동으로 재시작하게 함
    uvicorn.run("main:app", reload=True)