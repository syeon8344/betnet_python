# Flask 모듈 가져오기
from flask import Flask
from flask_cors import CORS


app = Flask(__name__)  # Flask 객체 생성
CORS(app)


from src.controller.SurveyController import *


# Flask 웹 실행
if __name__ == "__main__":
    app.run(debug=True)
