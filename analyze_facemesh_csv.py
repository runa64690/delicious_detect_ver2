import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 設定
# =========================
CSV_PATH = "./data/facemesh_landmarks.csv"

# 解析したい記録を指定
# None にすると、CSV内の最後の session_id / recording_id を自動で使う
TARGET_SESSION_ID = None
TARGET_RECORDING_ID = None

# 見たいランドマーク
TARGET_LANDMARKS = [61, 291, 216, 436, 13, 14]

LANDMARK_NAMES = {
    168: "nose_root",
    61: "left_mouth_corner",
    291: "right_mouth_corner",
    13: "upper_lip",
    14: "lower_lip",
    216: "left_cheek",
    436: "right_cheek",
}


# =========================
# CSV読み込み
# =========================
df = pd.read_csv(CSV_PATH)

if df.empty:
    print("CSVが空です")
    exit()

print("読み込んだ行数:", len(df))

# =========================
# 解析対象の session_id / recording_id を決める
# =========================
if TARGET_SESSION_ID is None:
    TARGET_SESSION_ID = df["session_id"].iloc[-1]

df_session = df[df["session_id"] == TARGET_SESSION_ID]

if TARGET_RECORDING_ID is None:
    TARGET_RECORDING_ID = df_session["recording_id"].iloc[-1]

df_trial = df[
    (df["session_id"] == TARGET_SESSION_ID) &
    (df["recording_id"] == TARGET_RECORDING_ID)
].copy()

if df_trial.empty:
    print("指定した session_id / recording_id のデータがありません")
    exit()

print("解析対象:")
print("  session_id:", TARGET_SESSION_ID)
print("  recording_id:", TARGET_RECORDING_ID)
print("  行数:", len(df_trial))

# 時間を0秒始まりにする
df_trial["t"] = df_trial["recording_elapsed_sec"]


# =========================
# 1. x座標の時系列
# =========================
plt.figure()

for landmark_id in TARGET_LANDMARKS:
    d = df_trial[df_trial["landmark_id"] == landmark_id]
    if d.empty:
        continue

    label = f"{landmark_id}: {LANDMARK_NAMES.get(landmark_id, '')}"
    plt.plot(d["t"], d["x_norm_by_eye"], label=label)

plt.xlabel("Time [s]")
plt.ylabel("x position normalized by eye distance")
plt.title("Horizontal movement")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 2. y座標の時系列
# =========================
plt.figure()

for landmark_id in TARGET_LANDMARKS:
    d = df_trial[df_trial["landmark_id"] == landmark_id]
    if d.empty:
        continue

    label = f"{landmark_id}: {LANDMARK_NAMES.get(landmark_id, '')}"
    plt.plot(d["t"], d["y_norm_by_eye"], label=label)

plt.xlabel("Time [s]")
plt.ylabel("y position normalized by eye distance")
plt.title("Vertical movement")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 3. 顔上での軌跡
# =========================
plt.figure()

for landmark_id in TARGET_LANDMARKS:
    d = df_trial[df_trial["landmark_id"] == landmark_id]
    if d.empty:
        continue

    label = f"{landmark_id}: {LANDMARK_NAMES.get(landmark_id, '')}"

    plt.plot(
        d["x_norm_by_eye"],
        d["y_norm_by_eye"],
        marker="o",
        markersize=2,
        label=label
    )

plt.xlabel("x position normalized by eye distance")
plt.ylabel("y position normalized by eye distance")
plt.title("Landmark trajectories around nose origin")
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()


# =========================
# 4. 開始時からの移動量
# =========================
summary_rows = []

for landmark_id in TARGET_LANDMARKS:
    d = df_trial[df_trial["landmark_id"] == landmark_id].copy()

    if d.empty:
        continue

    d = d.sort_values("t")

    x0 = d["x_norm_by_eye"].iloc[0]
    y0 = d["y_norm_by_eye"].iloc[0]

    dx = d["x_norm_by_eye"] - x0
    dy = d["y_norm_by_eye"] - y0

    movement = (dx ** 2 + dy ** 2) ** 0.5

    summary_rows.append({
        "landmark_id": landmark_id,
        "landmark_name": LANDMARK_NAMES.get(landmark_id, ""),
        "max_movement": movement.max(),
        "mean_movement": movement.mean(),
        "final_dx": dx.iloc[-1],
        "final_dy": dy.iloc[-1],
    })

summary_df = pd.DataFrame(summary_rows)

print("\n移動量の要約")
print(summary_df)


# =========================
# 5. 移動量を棒グラフで表示
# =========================
plt.figure()

labels = [
    f"{row.landmark_id}\n{row.landmark_name}"
    for row in summary_df.itertuples()
]

plt.bar(labels, summary_df["max_movement"])
plt.xlabel("Landmark")
plt.ylabel("Max movement normalized by eye distance")
plt.title("Maximum movement from start position")
plt.grid(True, axis="y")
plt.show()