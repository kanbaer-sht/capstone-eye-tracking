import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import random, string, time, requests, json, math
import tensorflow as tf
import numpy as np
import cv2
import threading

from aiohttp import web
from av import VideoFrame

from tensorflow import keras
from gaze_tracking import GazeTracking
from multiprocessing import Process, Queue
import multiprocessing

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

"""
============================================================================================================================================
"""

Std_INFO = {
    "test_id":30,
    "s_email":"1"
}

res = {
    "s_email":"",                               # 학생 고유 키 값
    "eye_caution":0                           # 부정행위 감지 카운트
}

# python <-> node.js 간 post 통신에 필요한 헤더 값
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

faceFlag = True
trackingFlag = True
flag = False
count = 0
URL_EYE = "http://3.89.30.234:3000/eyetracking"
ang1, ang2 = 0, 0
eye_caution = 0
size = [480, 640, 3]            # [0]:height, [1]:width, [2]:rgb

model_points = np.array([
        (0.0, 0.0, 0.0),  # 코
        (0.0, -330.0, -65.0),  # 턱
        (-225.0, 170.0, -135.0),  # 왼쪽 눈 끝
        (225.0, 170.0, -135.0),  # 오른쪽 눈 끝
        (-150.0, -150.0, -125.0),  # 입 왼쪽 끝
        (150.0, -150.0, -125.0)  # 입 오른쪽 끝
    ])

# 카메라 정보
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)  # width / 2, height / 2
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)


# 부정행위 카운트
def state():
    global count
    if flag:
        count += 1
    threading.Timer(1, state).start()

def face():
    global faceFlag, trackingFlag
    if faceFlag:
        faceFlag = False
        print('change to false')
    else:
        faceFlag = True
        trackingFlag = True
        print('change to true')
    threading.Timer(0.25, face).start()

class FaceDetector:

    def __init__(self,
        dnn_proto_text='models/deploy.prototxt',
        dnn_model='models/res10_300x300_ssd_iter_140000.caffemodel'):

        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        얼굴에 사각형 박스 구하기
        """
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        # 이미지에서 읽어온 결과 표시
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


class MarkDetector:
    # 얼굴 랜드마크

    def __init__(self, saved_model='models/pose_model'):

        self.face_detector = FaceDetector()
        self.cnn_input_size = 128
        self.marks = None

        # 저장한 모델파일로 부터 얼굴 랜드마크 모델 복구
        self.model = keras.models.load_model(saved_model)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        # 이미지에 사각형 박스 그리기
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)

    @staticmethod
    def move_box(box, offset):
        # 바라보는 방향에 맞추어 박스 설정
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):

        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:
            return box
        elif diff > 0:  # Height > width, 얇은 박스
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, 낮은 박스
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        #assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):

        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):

        _, raw_boxes = self.face_detector.get_faceboxes(
            image=image, threshold=0.5)
        a = []
        for box in raw_boxes:

            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                a.append(facebox)

        return a

    def detect_marks(self, image_np):

        predictions = self.model.signatures["predict"](
            tf.constant(image_np, dtype=tf.uint8))

        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):

        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 2, color, -1, cv2.LINE_AA)


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        color=(255, 255, 0), line_width=2):

    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size * 2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)


    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # 모든 선 그리기
    # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    k = (point_2d[5] + point_2d[8]) // 2
    # cv2.line(img, tuple(point_2d[1]), tuple(
    #     point_2d[6]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[2]), tuple(
    #     point_2d[7]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[3]), tuple(
    #     point_2d[8]), color, line_width, cv2.LINE_AA)

    return (point_2d[2], k)

"""
==========================================================================================================================================================================================
"""

def eyetracking(frame):

    global count, flag, ang1, ang2, eye_caution, faceFlag, trackingFlag

    video = frame

    if trackingFlag:
        gaze.refresh(video)
        #img = gaze.annotated_frame()
        text = ""

        if gaze.is_right():
            text = "Looking Right"
        elif gaze.is_left():
            text = "Looking Left"
        elif gaze.is_bottom():
            text = "Looking Bottom"
        elif gaze.is_top():
            text = "Looking Top"
        elif gaze.is_center():
            text = "Looking Center"
        print(text)
        cv2.putText(video, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


        faceboxes = mark_detector.extract_cnn_facebox(video)
        for facebox in faceboxes:
            face_img = video[facebox[1]: facebox[3],
                       facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (128, 128))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks([face_img])
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            shape = marks.astype(np.uint)

            # 얼굴 점 찍기
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))

            image_points = np.array([
                shape[30],  # 코끝
                shape[8],  # 턱
                shape[36],  # 왼쪽 눈 왼쪽 끝
                shape[45],  # 오른쪽 눈 오른쪽 끝
                shape[48],  # 입술 왼쪽
                shape[54]  # 입술 오른쪽
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)

            # for p in image_points:
            # cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            # cv2.putText(img, str(p), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = draw_annotation_box(video, rotation_vector, translation_vector, camera_matrix)

            # 선 표시
            # cv2.line(img, p1, p2, (0, 255, 255), 2)
            # cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

            # x축 각도 계산
            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            # y축 각도 계산
            try:
                m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1 / m)))
            except:
                ang2 = 90

            # x축
            cv2.putText(video, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            # y축
            cv2.putText(video, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

        if int(ang1) > 35 or int(ang1) < -35 or int(ang2) > 30:
            flag = True
        #elif text == "Looking Center":
        #    count = 0
        #    flag = False
        else:
            flag = True

        if count >= 5:
            video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
            #response = requests.post(url=URL_EYE, data=json.dumps(Std_INFO), headers=headers)
            #res = json.loads(response.text)
            #print(res)
            eye_caution += 1
            print(eye_caution)
            count = 0

        cv2.imshow('test', video)
        cv2.waitKey(1) & 0xFF
        trackingFlag = False


"""
==========================================================================================================================================================================================
"""

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """
    kind = "video"


    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        global faceFlag, trackingFlag

        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        rows = img.shape[0]
        height = img.shape[1]
        rgb = img.shape[2]
        test = np.full((rows, height, rgb), img, np.uint8)

        eyetracking(test)
        cv2.imshow('janus', test)
        cv2.waitKey(1) & 0xFF
        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):

        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            local_video = VideoTransformTrack(
                track
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    mark_detector = MarkDetector()
    # cap = cv2.VideoCapture(0)
    # ret, img = cap.read()
    # size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    gaze = GazeTracking()
    state()
    face()

    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )