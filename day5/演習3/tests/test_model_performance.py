import time
import joblib
import numpy as np

def test_model_accuracy():
    # モデルとテストデータのパスは適宜修正a
    model = joblib.load("../../models/latest_model.pkl")
    X_test = np.load("../../data/X_test.npy")
    y_test = np.load("../../data/y_test.npy")
    acc = model.score(X_test, y_test)
    assert acc > 0.8  # 目標精度を設定

def test_model_inference_time():
    model = joblib.load("../../models/latest_model.pkl")
    X_test = np.load("../../data/X_test.npy")
    start = time.time()
    _ = model.predict(X_test[:10])
    elapsed = time.time() - start
    assert elapsed < 1.0  # 10件で1秒以内など