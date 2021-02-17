import cv2
from gaze_tracking import GazeTracking
import dlib
import math
import threading
import time
import matplotlib


"""
GazeTracking
cv2로 웹캠 캡쳐
얼굴 모델링 감지
얼굴 모델링 데이터 가져오기
"""

flag    = False
count   = 0

# 부정행위 카운트
def state():
    global count
    if flag:
        count += 1
        print(count)
    else:
        print(count)
    threading.Timer(1, state).start()

gaze    = GazeTracking()
webcam  = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
state()

while True:
    rc, lc, ruc, luc, lruc = 0, 0, 0, 0, 0

    _, frame = webcam.read()        # 웹캠에서 새로 프레임 따오기
    gaze.refresh(frame)             # 프레임 갱신



    # 웹캠 영상 흑백 전환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 눈에 십자가 표시
    frame = gaze.annotated_frame()
    text = ""

    # 웹캠에서 얼굴 감지
    faces = detector(gray)

    # facelandmark를 웹캠 영상에 그리기
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        # 각 점 위치마다 그리기
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # 고개돌아감을 감지하기 위한 얼굴의 각 좌표 값 저장
            # 0 : 얼굴 오른쪽, 36 : 오른쪽 눈
            # 16 : 얼굴 왼쪽, 45 : 왼쪽 눈
            # 17 : 오른쪽 눈썹 끝, 26 : 왼쪽 눈썹 끝
            if n == 0:
                rx1 = x
                ry1 = y
            if n == 36:
                rx2 = x
                ry2 = y
            if n == 16:
                lx1 = x
                ly1 = y
            if n == 45:
                lx2 = x
                ly2 = y
            if n == 17:
                rbx = x
                rby = y
            if n == 26:
                lbx = x
                lby = y

            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # 두 점 사이의 거리를 계산
        rx3 = rx2 - rx1
        ry3 = ry2 - ry1
        lx3 = lx2 - lx1
        ly3 = ly2 - ly1
        rux = rbx - rx1
        ruy = rby - ry1
        lux = lbx - lx1
        luy = lby - ly1
        rc = math.sqrt((rx3 * rx3) + (ry3 * ry3))
        lc = math.sqrt((lx3 * lx3) + (ly3 * ly3))
        ruc = math.sqrt((rux * rux) + (ruy * ruy))
        luc = math.sqrt((lux * lux) + (luy * luy))
        lruc = math.sqrt((rux * rux) + (lux * lux))

    """
    ============================================================================
    """

    # 얼굴이 감지 된 경우
    if faces:
        # 왼쪽으로 고개 돌리는 경우
        if rc > 65 and lc < 30:
            cv2.putText(frame, "Left!!!", (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
            flag = True
        # 오른쪽으로 고개 돌리는 경우
        elif rc < 30 and lc > 65:
            cv2.putText(frame, "Right!!!", (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
            flag = True
        # 위를 보는 경우
        elif ruc > 35 or luc > 35:
            if rc > 43 and lc > 43:
                cv2.putText(frame, " Up!!!", (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
                flag = True
        else:
            flag = False
            count = 0
        """
        # 밑을 보는 경우
        elif ruc < 24 or luc < 24:
            #if rc > 40 and lc > 40:
                cv2.putText(frame, "Down!!!", (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
                flag = True
        """
    # 얼굴이 감지되지 않은 경우
    else:
        cv2.putText(frame, "No Faces!!!", (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
        flag = True

    if count >= 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count = 0

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

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break