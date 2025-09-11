import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import pygame, time, math


CSV_FILE = "lidar_data.csv"

df = pd.read_csv(CSV_FILE, header=None, names=["reading", "label"], dtype=str, engine='python')
df = df[df["label"] != "U"].reset_index(drop=True)

print("Dataset size: {}".format(len(df)))
print("Example row reading: {}...".format(df.iloc[0, 0][:120]))
print("Example label: {}".format(df.iloc[0, 1]))

# parse frames
def parse_frame(row_str):
    return np.asarray([float(val) for val in str(row_str).split(";") if val.strip() != ""], dtype=float)


frames = [parse_frame(r) for r in df["reading"]]
label = df["label"].values
print("Frames parsed: {}".format(len(frames)))
print("Example frame shape: {}".format(frames[0].shape))
print("Label distribution: {}\n".format(pd.Series(label).value_counts()))

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

frames_ds = [uniform_sample(f, target=200) for f in frames]
x = np.array(frames_ds)
y = label
print("Freature matrix shape: {}".format(x.shape))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print("Train shape: {} Test shape: {}".format(x_train.shape, x_test.shape))


# balanced training set
train_df = pd.DataFrame(x_train)
train_df["label"] = y_train
min_count = train_df["label"].value_counts().min()

balanced_train = pd.concat([resample(train_df[train_df["label"] == label], replace=False, n_samples=min_count, random_state=42)for label in train_df["label"].unique()])

balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

x_train_balanced = balanced_train.drop('label', axis=1).values
y_train_balanced = balanced_train["label"].values

print("Balanced train shape: {}".format(x_train_balanced.shape))
print("Label distribution: {}\n".format(pd.Series(y_train_balanced).value_counts()))

# feature selection

selector = VarianceThreshold(threshold=1e-4)
x_train_var = selector.fit_transform(x_train_balanced)
x_test_var = selector.transform(x_test)

k = 20
kbest = SelectKBest(score_func=f_classif, k=k)
x_train_reduced = kbest.fit_transform(x_train_var, y_train_balanced)
x_test_reduced = kbest.transform(x_test_var)

selected_idx = kbest.get_support(indices=True)

print("Reduced feature shape: {} - {}".format(x_test_reduced.shape, x_test_reduced.shape))
print("Selected feature indexes: {}".format(selected_idx))

# visualization 

def visualize_frames(frames, highlight_idx=None, delay=0.05):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    center = (300, 300)
    scale = 400.0
    idx = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running =False
            
            screen.fill((240, 240, 240))
            pygame.draw.circle(screen, (50, 50, 255), center, 20)

            frame = frames[idx % len(frames)]
            n_point = len(frame)
            for i, dist in enumerate(frame):
                angle = (i / n_point) * 2 * math.pi
                px = int(center[0] + math.cos(angle) * dist * scale)
                py = int(center[1] + math.sin(angle) * dist * scale)
                color = (255, 0, 0) if highlight_idx is not None and i in highlight_idx else (20, 20, 20)

                pygame.draw.circle(screen, color, (px, py), 2)

            pygame.display.flip()
            time.sleep(delay)
            idx += 1
    
    pygame.quit()

# visualize_frames(frames_ds, highlight_idx=selected_idx)

# train and evaluate classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

clf = RandomForestClassifier(n_estimators=20, max_depth=None, random_state=41, n_jobs=1)
clf.fit(x_train_reduced, y_train_balanced)

y_pred = clf.predict(x_test_reduced)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy: {}".format(acc))

print("Classification Report: \n{}".format(classification_report(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# export trained model

import joblib

MODEL_FILE = "/home/omni/Project/jetbot/lidar_classifer.joblib"
selector_file = "/home/omni/Project/jetbot/selector.joblib"
kbest_file = "/home/omni/Project/jetbot/kbest.joblib"

joblib.dump(clf, MODEL_FILE)
joblib.dump(selector, selector_file)
joblib.dump(kbest, kbest_file)

print("Random Forrest model exported to {}".format(MODEL_FILE))