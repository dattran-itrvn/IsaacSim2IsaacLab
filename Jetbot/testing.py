from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import joblib
import pandas as pd
import numpy as np

model_file = "/home/omni/Project/jetbot/lidar_classifer.joblib"
selector_file = "/home/omni/Project/jetbot/selector.joblib"
kbest_file = "/home/omni/Project/jetbot/kbest.joblib"

# uniform downsampling rate
def uniform_sample(points, target=200):
    n = len(points)
    if n == 0:
        return np.zeros(target)
    
    if n <= target:
        out = np.zeros(target)
        out[:n] = points
        return out

    idx = np.linspace(0, n - 1, target, dtype=int)
    return points[idx]

selector = joblib.load(selector_file)
kbest = joblib.load(kbest_file)
clf = joblib.load(model_file)

CSV_FILE = "lidar_data.csv"
df = pd.read_csv(CSV_FILE, header=None, names=["reading", "label"], dtype=str, engine='python')
df = df[df["label"] != "U"].reset_index(drop=True)
# parse frames
def parse_frame(row_str):
    return np.asarray([float(val) for val in str(row_str).split(";") if val.strip() != ""], dtype=float)


frames = [parse_frame(r) for r in df["reading"]]
frames_ds = [uniform_sample(f, target=200) for f in frames]
x = np.array(frames_ds)

x_test_var = selector.transform(x[0].reshape(1, -1))
x_test_reduced = kbest.transform(x_test_var)
y_pred = clf.predict(x_test_reduced)
print(y_pred)