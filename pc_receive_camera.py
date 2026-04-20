import socket
import struct
import pickle
import cv2

HOST = "0.0.0.0"
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("接続待機中...")
conn, addr = server_socket.accept()
print("接続:", addr)

data = b""
payload_size = struct.calcsize("Q")

try:
    while True:
        # サイズ情報を受信
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < payload_size:
            break

        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_size)[0]

        # 画像データ本体を受信
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        frame_data = data[:msg_size]
        data = data[msg_size:]

        encoded = pickle.loads(frame_data)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        cv2.imshow("Raspberry Pi Camera", frame)

        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()