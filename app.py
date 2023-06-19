from flask import Flask, request, jsonify
import joblib
import pickle
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # JSON 데이터 받기
    data = request.get_json()
    # 데이터 전처리
    df = pd.DataFrame([data])

    # 예측 수행
    predictions = np.round(model.predict(df))

    # 예측 결과 반환
    response = {'해당 국가에서 발간한 AI 정책 수(예측)': predictions.tolist()}

    return jsonify(response)

@app.errorhandler(500)
def handle_internal_server_error(e):
    error_message = "Internal Server Error"
    # 추가적인 오류 처리 로직을 구현할 수 있습니다.
    # 예를 들어, 로그 기록, 오류 알림 등을 수행할 수 있습니다.
    return jsonify(error=error_message), 500

if __name__ == '__main__':
    app.run(debug=True)

