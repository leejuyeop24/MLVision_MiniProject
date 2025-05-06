import joblib

# 예: 이미 학습된 모델 (xgb_model)을 저장
joblib.dump(xgb_model, 'xgb_model.pkl')   # 파일명은 원하는 대로 설정 가능
print("모델이 xgb_model.pkl 로 저장되었습니다.")
