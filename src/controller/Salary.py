from flask import request ,jsonify
from src.app import app
from src.service.Playercrawl import *

@app.route("/salary",methods=['GET'])
def getSalary():
    name = request.args.get('name',type=str)
    print(name)
    csv_path= 'crawl_csv/stat2024.csv'
    df=pd.read_csv(csv_path)
    # 이름에 맞는 행 필터링
    filtered_rows = df[df['선수명'] == name]

    # 결과가 없으면 빈 리스트 반환
    if filtered_rows.empty:
        return jsonify([])  # 빈 리스트로 응답

    # JSON 형태로 결과 반환
    return jsonify(filtered_rows.to_dict(orient='records'))
