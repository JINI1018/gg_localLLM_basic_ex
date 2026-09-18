from urllib import request

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from sqlalchemy.orm import Session
import uvicorn
from fastapi.responses import RedirectResponse

from database import engine, SessionLocal, Base
import models 

# models에 정의한 모든 클래스, 연결한 DB엔진에 테이블로 생성
Base.metadata.create_all(bind=engine)

# FastAPI() 객체 생성
app = FastAPI()

# 현재 파일의 절대 경로를 구함
# __file__은 현재 파일 경로, os.path.realpath(__file__)은 절대 경로, os.path.dirname()은 디렉토리 경로를 구함
# 따라서 abs_path는 현재 파일이 속한 디렉토리의 절대 경로가 됨
# 예: /home/user/project/todos_app
abs_path = os.path.dirname(os.path.realpath(__file__))
# print(abs_path)

# templates 폴더 식별 객체 설정
# Jinja2Templates 객체를 생성하여 HTML 템플릿을 렌더링할 준비를 함
# templates = Jinja2Templates(directory="templates")
templates = Jinja2Templates(directory=f"{abs_path}/templates")

# static 폴더(정적파일 폴더)를 fastAPI에서 인식할 수 있도록 마운트시킴
# app.mount("/static", StaticFiles(directory=f"static"), name="static")
app.mount("/static", StaticFiles(directory=f"{abs_path}/static"), name="static")

def get_db():
    db = SessionLocal()
    try:
        yield db # 처리가 끝날 때까지 기다려라.
    finally:
        # 마지막에 무조건 닫음
        db.close()

# http://localhost:8000/ = 루트 주소('/')에 GET 요청이 오면 home 함수 실행
@app.get("/")
def home(request: Request,
         db_ss: Session = Depends(get_db)
         ):
    # 테이블 조회
    # 비즈니스 로직 처리
    # 예: todo 개수를 계산하거나 데이터베이스에서 가져오는 로직
    todos_list = db_ss.query(models.Todo).order_by(models.Todo.id.desc()).all()
    # print(todos_list)
    # for todo in todos_list:
    #     print(f"{todo.id}, {todo.task}")

    return templates.TemplateResponse(
        request = request,
        name = "index.html",
        context={"todos": todos_list}
    )

# todo 데이터를 받아서 DB에 저장하기 위한 POST 요청 처리
# http://127.0.0.1:8000/add/
@app.post("/add")
def add(request: Request,
        task: str = Form(...), 
        db_ss: Session = Depends(get_db)):
    # 클라이언트에서 textarea에서 입력 데이터 넘어온것 확인
    print(task)
    # task 데이터를 받고, todo class를 통해서, 테이블과 연결된 객체 생성
    todo = models.Todo(task=task)
    # todos 테이블에 task 추가
    db_ss.add(todo)

    # 테이블에 반영(commit)
    db_ss.commit()

    # 엔드포인트 함수 home으로 redirect
    return RedirectResponse(url=app.url_path_for("home"),
                            status_code=status.HTTP_303_SEE_OTHER)

# todo 수정할 레코드 조회
@app.get("/edit/{todo_id}")
def edit(request: Request,
         todo_id: int,
         db_ss: Session = Depends(get_db)):

    # 수정 요청 id로 todo 조회
    todo = db_ss.query(models.Todo).filter(models.Todo.id == todo_id).first()
    print(todo.task)

    # 예외처리
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    # 조회 결과 edit.html에 랜더링 해서 응답하기
    return templates.TemplateResponse(
        request=request,
        name="edit.html",
        context={"todo": todo}
    )

# todo 수정할 내용 반영하기
@app.post("/edit/{todo_id}")
def update(request: Request,
         todo_id: int,
         task: str = Form(...),
         completed: bool = Form(False),
         db_ss: Session = Depends(get_db)):

    todo = db_ss.query(models.Todo).filter(models.Todo.id == todo_id).first()
    todo.task = task
    todo.completed = completed
    db_ss.commit()

    return RedirectResponse(url=app.url_path_for("home"),
                            status_code=status.HTTP_303_SEE_OTHER)

# todo 삭제 확인 화면
@app.get("/delete/{todo_id}")
def delete_confirm(request: Request,
                   todo_id: int,
                   db_ss: Session = Depends(get_db)):
    todo = db_ss.query(models.Todo).filter(models.Todo.id == todo_id).first()

    # 예외처리: 해당 id의 todo가 존재하지 않으면 404 에러 발생
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")

    # todo가 존재하면 삭제 확인 페이지 랜더링
    return templates.TemplateResponse(
        request=request,
        name="delete.html",
        context={"todo": todo}
    )


# todo 삭제하기
@app.post("/delete/{todo_id}")
def delete(todo_id: int,
           db_ss: Session = Depends(get_db)):
    # 삭제 요청 id로 todo 조회
    todo = db_ss.query(models.Todo).filter(models.Todo.id == todo_id).first()

    # 예외처리
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")

    # todo 삭제
    db_ss.delete(todo)
    db_ss.commit()
    return RedirectResponse(url=app.url_path_for("home"),
                            status_code=status.HTTP_303_SEE_OTHER)


# uv run fastapi dev
if __name__ == "__main__":
    # uvicorn.run("main:app", reload=True)
    # # main은 현재_파일 이름: FastAPI 객체_식별자.
    # reload=True: 개발자 모드 실행. 코드 변경 시 서버를 자동으로 재시작하게 함
    uvicorn.run("main:app", reload=True)
