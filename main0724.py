import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Streamlitアプリのタイトルを設定
st.title("YOLOv8 物体検知アプリ")

# 動画ファイルのアップロード
video_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4"])

# 動画の表示
if video_file is not None:
    # 動画ファイルを保存
    video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # 動画を表示
    st.video(video_path)

    # 物体検知ボタンが押されたら実行
    if st.button("物体検知を実行"):
        # YOLOv8で物体検知
        model = YOLO('yolov8n.pt')
        results = model(video_path, vid_stride=10, stream=True, classes=0)

        # 保存先のファイルパス
        save_path = "./output.mp4"

        # 動画を保存する処理
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = None

        # フレームレートを取得
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # フレームごとの検知結果を処理
        person_counts = []
        timestamps = []
        for frame_num, result in enumerate(results):
            # フレームを取得
            frame = result.orig_img

            # 検出結果を描画
            person_count = 0
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = box
                label = f"{result.names[int(cls.item())]} {conf.item():.2f}"
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # RGB color for rectangle
                frame = cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # RGB color for text
                if result.names[int(cls.item())] == "person":
                    person_count += 1

            # 現在の時間（秒）を計算
            timestamp = frame_num / fps

            # フレームごとの人数と時間を記録
            person_counts.append(person_count)
            timestamps.append(timestamp)

            if video_writer is None:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(save_path, fourcc, 30, (width, height))

            video_writer.write(frame)

        # ファイルを閉じる
        video_writer.release()

        # 検知結果の動画を表示
        if os.path.exists(save_path):
            st.write(f"動画ファイルが存在します。サイズは {os.path.getsize(save_path)} バイトです。")
            st.video(save_path)
        else:
            st.write("動画ファイルが存在しません。")

        # 一時ファイルを削除
        os.remove(video_path)

        # プロットを作成
        plt.plot(timestamps, person_counts)
        plt.xlabel("Time (s)")
        plt.ylabel("Person Count")
        plt.title("Person Count over Time")
        st.pyplot(plt)