# day08 > 1_플라스크.py
"""
    Flask
        - 파이썬으로 만들어진 웹 프레임워크 ( vs JAVA Spring)
        - Flask vs Django
    - 설치
        1) Flask 모듈 설치
            방법 1: from flask에 커서 -> 빨간 느낌표 -> install package
            방법 2: 메뉴 > 파일 > 설정 > 사이드바: 프로젝트 > 인터프리터 > [+] alt+insert
        2) Flask 모듈 가져오기
            from flask import Flask
        3) Flask 객체 생성
            app = Flask(__name__)
        4) Flaks 프레임워크 실행
            if __name__ == "__main__":
                app.run()

                45245
"""
# Flask 모듈 가져오기
from flask import Flask
# Flask 객체 생성
app = Flask(__name__)

#
# HTTP GETmapping 설정
@app.route('/')
def index():  # 매핑 함수
    return "Hello Flask!"


# Flask 웹 실행
if __name__ == "__main__":
    app.run(debug=True)

# 콘솔 확인
# Flask port: 5000 -> http://localhost:5000
# 테스트 1: 크롬 주소창, 테스트 2: Talend API, 테스트 3: JS AJAX
"""
 * Serving Flask app '1_플라스크'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 132-514-738
127.0.0.1 - - [23/Aug/2024 10:16:49] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [23/Aug/2024 10:16:49] "GET /favicon.ico HTTP/1.1" 404 -
"""
"test"