from flask import request, abort, jsonify
from src.app import app
import pandas as pd
import json

# 월간 경기일정을 CSV에서 읽어와서 (인덱스 포함) JSON형태의 문자열로 보내기
# 연도와 월 포함시 특정 월 파일 정보 제공, 기본값은 현재 날짜
@app.route('/monthlyschedule', methods=['GET'])
def monthly_schedule():
    # 쿼리 문자열에서 year와 month 가져오기
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)

    if year and month:
        date = f'{year:04d}{month:02d}'
    else:
        date = pd.to_datetime('today').strftime('%Y%m')  # 날짜 입력이 없을 시 현재 년도 월
    try:
        df = pd.read_csv(f'crawl_csv/monthly_schedule/월간경기일정_{date}.csv', encoding='utf-8', dtype=str)
    except FileNotFoundError:
        print('/monthlyschedule: 해당 연월의 월간경기일정 파일이 없습니다.')
        return abort(404)  # 404 Not Found 응답 반환
    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    # jsonify() vs json.dumps(): jsonify()는 content-type: application/json; charset=UTF-8 헤더를 자동으로 추가해준다
    # 혹시 모를 스크립트 공격 예방을 위해 적절한 탈출문자 처리도 되므로 jsonify()가 선호된다
    # {\"월\":\"09\",\"일\":\"01\",\"시작시간\":\"14:00\",\"어웨이팀명\":\"롯데\",\"홈팀명\":\"두산\",\"어웨이점수\":\"4\",\"홈점수\":\"3\",\"비고\":\"-\",\"경기코드\":\"20240907-KIA-1700\"}}
    return jsonify(json.loads(df.to_json(orient='records', force_ascii=False)))


# index.html 이틀치 경기일정 가져오기
# # 월간 경기일정을 CSV에서 읽어와서 (인덱스 포함) JSON형태의 문자열로 보내기
# # 연도, 월, 일을 받아 특정 오늘/내일/그다음날... 에서 순서대로 2일치 결과 전송
@app.route('/getschedule', methods=['GET'])
def get_schedule():
    # 쿼리 문자열에서 year와 month 가져오기
    date = request.args.get('date', type=str)
    year, month, day = date.split('-')  # date -> year, month, day

    csv_date = f'{year}{month}'

    df = pd.read_csv(f'crawl_csv/monthly_schedule/월간경기일정_{csv_date}.csv', encoding='utf-8', dtype={"월": str, "일": str})

    # 기준 날짜 설정
    start_date = pd.to_datetime(f"{year}-{month}-{day}")

    # 기준 날짜 이후 데이터 필터링
    df['날짜'] = pd.to_datetime(year + '-' + df['월'] + '-' + df['일'])
    filtered_df = df[df['날짜'] >= start_date]

    # 날짜별로 행이 있는지 확인
    date_counts = filtered_df['날짜'].value_counts()
    available_dates = date_counts[date_counts > 0].index

    # 결과 DataFrame 생성
    result = []
    for date in available_dates[:2]:  # 이틀치만 가져오기
        result.append(filtered_df[filtered_df['날짜'] == date])

    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    # jsonify() vs json.dumps(): jsonify()는 content-type: application/json; charset=UTF-8 헤더를 자동으로 추가해준다
    # 혹시 모를 스크립트 공격 예방을 위해 적절한 탈출문자 처리도 되므로 jsonify()가 선호된다
    # {\"월\":\"09\",\"일\":\"01\",\"시작시간\":\"14:00\",\"어웨이팀명\":\"롯데\",\"홈팀명\":\"두산\",\"어웨이점수\":\"4\",\"홈점수\":\"3\",\"비고\":\"-\",\"경기코드\":\"20240907-KIA-1700\"}}
    if result:
        result_df = pd.concat(result)  # 이틀치 DataFrame 합치기
        return jsonify(result_df.to_json(orient='records', force_ascii=False))  # JSON 형태로 반환
    else:
        return jsonify([])  # 데이터가 없을 경우 빈 리스트 반환


# 모든 연도 타자 크롤링된 데이터 내보내기
@app.route('/gethittertable', methods=['GET'])
def get_hitter_table():
    # 쿼리 문자열에서 연도 가져오기, 기본값은 2024
    year = request.args.get('year', default=2024, type=int)
    # 파일 읽기, 파일 유효성 체크 포함
    try:
        df = pd.read_csv(f'crawl_csv/hitter/팀기록_타자_{year}.csv', encoding='utf-8')
    except FileNotFoundError:
        print('/gethittertable: 해당 연도 타자 파일이 없습니다.')
        return abort(404)  # 404 Not Found 에러 반환
    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    return jsonify(df.to_json(orient='records', force_ascii=False))


# 특정 연도 투수 크롤링된 데이터 내보내기
@app.route('/getpitchertable', methods=['GET'])
def get_pitcher_table():
    # 쿼리 문자열에서 연도 가져오기, 기본값은 2024
    year = request.args.get('year', default=2024, type=int)
    # 파일 읽기, 파일 유효성 체크 포함
    try:
        df = pd.read_csv(f'crawl_csv/pitcher/팀기록_투수_{year}.csv', encoding='utf-8')
    except FileNotFoundError:
        print('/getpitchertable: 해당 연도 투수 파일이 없습니다.')
        return abort(404)  # 404 Not Found 에러 반환
    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    return jsonify(df.to_json(orient='records', force_ascii=False))


# 특정 연도 주루 크롤링된 데이터 내보내기
@app.route('/getrunnertable', methods=['GET'])
def get_runner_table():
    # 쿼리 문자열에서 연도 가져오기, 기본값은 2024
    year = request.args.get('year', default=2024, type=int)
    # 파일 읽기, 파일 유효성 체크 포함
    try:
        df = pd.read_csv(f'crawl_csv/runner/팀기록_주루_{year}.csv', encoding='utf-8')
    except FileNotFoundError:
        print('/getrunnertable: 해당 연도 주루 파일이 없습니다.')
        return abort(404)  # 404 Not Found 에러 반환
    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    return jsonify(df.to_json(orient='records', force_ascii=False))


# 특정 연도 팀 순위 크롤링된 데이터 내보내기
@app.route('/getranktable', methods=['GET'])
def get_rank_table():
    # 쿼리 문자열에서 연도 가져오기, 기본값은 2024
    year = request.args.get('year', default=2024, type=int)
    # 파일 읽기, 파일 유효성 체크 포함
    try:
        df = pd.read_csv(f'crawl_csv/rank/팀순위_{year}.csv', encoding='utf-8')
    except FileNotFoundError:
        print('/getranktable: 해당 연도 팀 순위 파일이 없습니다.')
        return abort(404)  # 404 Not Found 에러 반환
    # DataFrame을 JSON 형태의 문자열로 변환해서 전송
    return jsonify(df.to_json(orient='records', force_ascii=False))
