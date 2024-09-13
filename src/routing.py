from app_start import app


@app.route('/test', methods=['GET'])
def index():
    return "Hello Flask!"


# 월간 경기일정을 CSV에서 읽어와서 (인덱스 포함) JSON형태의 문자열로 보내기
