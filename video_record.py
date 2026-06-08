import socket
import struct
import pickle
import cv2
import time
import os

HOST = "0.0.0.0"
PORT = 9999

SAVE_DIR = "video_data"
os.makedirs(SAVE_DIR, exist_ok=True)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("接続待機中...")
conn, addr = server_socket.accept()
print("接続:", addr)

data = b""
payload_size = struct.calcsize("Q")

recording = False
video_writer = None
record_start_time = None
video_count = 1

# 既存ファイルがある場合、次の番号から保存
while os.path.exists(os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")):
    video_count += 1

try:
    while True:
        # サイズ情報を受信
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("接続が切断されました")
            data += packet

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_size)[0]

        # 画像データ本体を受信
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                raise ConnectionError("接続が切断されました")
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        encoded = pickle.loads(frame_data)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        display_frame = frame.copy()

        # 録画中なら動画に保存
        if recording:
            video_writer.write(frame)

            elapsed_time = time.time() - record_start_time
            timer_text = f"REC {elapsed_time:.1f} sec"

            cv2.putText(
                display_frame,
                timer_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        cv2.imshow("Raspberry Pi Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            if not recording:
                h, w, _ = frame.shape

                filename = os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = 10.0

                video_writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    fps,
                    (w, h)
                )

                recording = True
                record_start_time = time.time()

                print(f"録画開始: {filename}")

        elif key == ord("e"):
            if recording:
                recording = False
                record_start_time = None

                video_writer.release()
                video_writer = None

                print(f"録画終了: video_{video_count:03d}.mp4")

                video_count += 1
                while os.path.exists(os.path.join(SAVE_DIR, f"video_{video_count:03d}.mp4")):
                    video_count += 1

finally:
    if video_writer is not None:
        video_writer.release()

    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()