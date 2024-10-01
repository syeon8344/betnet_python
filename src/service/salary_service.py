import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from flask import request ,jsonify

def predictSalary(name):
    stat2023=pd.read_csv('crawl_csv/stat2023.csv')
    y_2023=['연봉']
    x_2023=stat2023.drop(['연봉'],axis=1,inplace=False)

    print( stat2023.info())
    print( x_2023.info() )
    print( y_2023.info() )


    x_features=stat2023['타석' ,'타율','홈런','루타','타점','도루','장타율','출루율'].values
    print( x_features )
    from sklearn.preprocessing import StandardScaler
    x_2023_s_scaled=StandardScaler().fit_transform(x_features)
    print( x_2023_s_scaled )


    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x_2023_s_scaled,y_2023,test_size=0.1,random_state=156)
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    y_predict=lr.predict(x_test)
    print(y_predict)
    print( y_train )

    mse=mean_squared_error(y_test,y_predict)
    rmse=np.sqrt(mse)
    print(f'MSE:{mse:.3f}, RMSE:{rmse:.3f}')
    print(f'R^2(Variance score): {r2_score(y_test,y_predict):.3f}')
    print('Y 절편 값: ',lr.intercept_)
    print('회귀 계수 값: ',np.round(lr.coef_,1)) # 기울기 값

    stat2024 =pd.read_csv('crawl_csv/stat2024.csv')
    filtered_rows = stat2024[stat2024['선수명'] == name]
    # 결과가 없으면 빈 리스트 반환
    if filtered_rows.empty:
        return jsonify([])  # 빈 리스트로 응답

    x_2024 = filtered_rows[['타석', '타율', '홈런', '루타', '타점', '도루', '장타율', '출루율']]
    x_2024_s_scaled=StandardScaler().fit_transform(x_2024)
    print( x_2024_s_scaled )
    y_predict=lr.predict(x_2024_s_scaled)
    print(y_predict)
    print(stat2024.head())
    print( x_2024.info() )
    print( x_2024.head() )
    # 예측 결과를 새로운 컬럼 '예상연봉'으로 추가
    filtered_rows['예상연봉'] = y_predict
    return jsonify(filtered_rows.to_dict(orient='records'))
