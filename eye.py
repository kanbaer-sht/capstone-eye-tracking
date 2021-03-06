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
import aiohttp
import dlib
import socketio
import numba
from numba import jit

from aiohttp import web
from av import VideoFrame

from tensorflow import keras
from gaze_tracking import *
from multiprocessing import Process, Queue
import multiprocessing

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

pcs = set()

# keras 코드 warning 숨기기
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
===================================================================================================================================================================================================
"""
std_list = []

std_dic = {

}

std_eye = {

}

Std_INFO = {

}


text = ""

sio = socketio.Client()
sio.connect('http://3.89.30.234:3001')
faceFlag = True
trackingFlag = True
flag = False
count = 0
ang1, ang2 = 0, 0
text = ""
p1 = 0
p2 = 0

# 3d 모델 각 점 위치
model_points = np.array([
        (0.0, 0.0, 0.0),  # 코
        (0.0, -330.0, -65.0),  # 턱
        (-225.0, 170.0, -135.0),  # 왼쪽 눈 끝
        (225.0, 170.0, -135.0),  # 오른쪽 눈 끝
        (-150.0, -150.0, -125.0),  # 입 왼쪽 끝
        (150.0, -150.0, -125.0)  # 입 오른쪽 끝
    ])


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
    else:
        faceFlag = True
        trackingFlag = True
    threading.Timer(0.25, face).start()

class Student:
    def __init__(self, s_number, test_id):
        self.s_number = s_number
        self.eye_caution = 0
        self.test_id = test_id

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

# 눈 방향 감지 함수
def eye(frame):
    global count, flag, ang1, ang2, faceFlag, trackingFlag, sio, text, gaze, font



    video = frame
    gaze.refresh(video)
    #video = gaze.annotated_frame()
    text = ""

    if gaze.is_top():
        text = "Top"
    elif gaze.is_right():
        text = "Right"
    elif gaze.is_left():
        text = "Left"
    elif gaze.is_center():
        text = "Center"

    #print(text)
    cv2.putText(video, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    return text


# 얼굴 인식 및 각도 측정 함수
def faceRecog(frame, camera_matrix):
    global ang1, ang2,mark_detector, font, p1, p2
    video = frame

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
        # mark_detector.draw_marks(video, marks, color=(0, 255, 0))

        image_points = np.array([
            shape[30],  # 코끝
            shape[8],  # 턱
            shape[36],  # 왼쪽 눈 왼쪽 끝
            shape[45],  # 오른쪽 눈 오른쪽 끝
            shape[48],  # 입술 왼쪽
            shape[54]  # 입술 오른쪽
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

        # (success, rotation_vector, translation_vector) = Process(target=cv2.solvePnP, args=(model_points, image_points, camera_matrix,dist_coeffs, cv2.SOLVEPNP_UPNP)).start()

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        # (nose_end_point2D, jacobian) = Process(target=cv2.projectPoints, args=(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,translation_vector, camera_matrix, dist_coeffs)).start()

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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
gaze = GazeTracking()
mark_detector = MarkDetector()

@sio.on('eyetrackingcount')
async def eyetrackingcount(msg):
    print(msg)

def eyetracking(frame, test_id, s_number, eye_count, eye_caution, size):
    global flag, ang1, ang2, faceFlag, trackingFlag, sio, text, font, Std_INFO

    # 카메라 정보
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)  # width / 2, height / 2
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    #print('eye_count : ',eye_count)
    #print('eye_caution : ',eye_caution)
    print(text)
    video = frame

    text = eye(frame)
    faceRecog(frame, camera_matrix)

    #cv2.imshow(str(s_number) + 'eye', video)
    #cv2.waitKey(1) & 0xFF

    # 3초 이상 바깥응시, 고개 돌아간 상황
    if eye_count >= 25 and eye_caution <= 4:
        print("부정행위가 발생했습니다!")
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        eye_caution += 1

        if eye_caution >= 0:
            Std_INFO = {
                'test_id':test_id,
                's_number':s_number
            }
            sio.emit("eyetracking", Std_INFO)
            print(s_number, ':', eye_caution)


            Std_INFO = {

            }

        #cv2.imshow(str(s_number) + 'eye', video)
        #cv2.waitKey(1) & 0xFF

        return "caution up"
    else:
        pass
        #cv2.imshow(str(s_number) + 'eye', video)
        #cv2.waitKey(1) & 0xFF
    # x축
    #cv2.putText(str(s_number) + 'janus', str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
    # y축
    #cv2.putText(str(s_number) + 'janus', str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
    # 부정행위가 발생했을 때, 시간 카운트 플래그 설정 후 리턴
    if int(ang1) > 35 or int(ang1) < -35 or int(ang2) > 30:
        flag = True
        return "count up"
    elif text == "Center":
        flag = False
        return "count reset"
    else:
        flag = True
        return "count up"
