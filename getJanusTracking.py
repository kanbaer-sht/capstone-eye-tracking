import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import random, string, time, requests, math
import tensorflow as tf
import numpy as np
import cv2
import threading
import aiohttp
import dlib
import socketio

from aiohttp import web
from av import VideoFrame

from tensorflow import keras
from gaze_tracking import *
from multiprocessing import Process, Queue
import multiprocessing
import eye

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

std_dic = {

}

std_eye = {

}

std_cnt = {
    "s_number":0,
    "count":0,
    "eye_caution":0
}

pcs = set()

count = 0
countFlag = True
faceFlag = True
trackingFlag = True
size_mat = [0,0,0]

# 부정행위 카운트
def state():
    global count
    if countFlag:
        count += 1
        print(count, countFlag)
    threading.Timer(1, state).start()

def face():
    global faceFlag, trackingFlag
    if faceFlag:
        faceFlag = False
    else:
        #faceFlag = True
        trackingFlag = True
    threading.Timer(0.25, face).start()

def transaction_id():
    return "".join(random.choice(string.ascii_letters) for x in range(12))

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """
    kind = "video"

    def __init__(self, track, s_number, test_id):
        super().__init__()  # don't forget this!
        self.track = track
        self.s_number = s_number
        self.test_id = test_id

    async def recv(self):

        global trackingFlag, count, countFlag, size_mat

        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")      # videoframe reformat to ndarray
        size_mat[0] = img.shape[0]                  # set size of frame to use openCV
        size_mat[1] = img.shape[1]
        size_mat[2] = img.shape[2]
        test = np.full(size_mat, img, np.uint8)     # ndarray to image data for openCV

        if trackingFlag and std_cnt["eye_caution"] <= 3:
            trackingFlag = False

            text = eye.eyetracking(frame=test, test_id=self.test_id, s_number=self.s_number, eye_count=std_cnt["count"],
                            eye_caution=std_cnt["eye_caution"], size=size_mat)

            print(std_cnt["count"])
            if text == "count up":
                std_cnt["count"] += 1
                print(std_dic)
                print(std_eye)
            elif text == "count reset":
                std_cnt["count"] = 0
                print(std_dic)
                print(std_eye)
            elif text == "caution up":
                std_cnt["count"] = 0
                std_eye[self.s_number]["eye_caution"] += 1
                std_cnt["eye_caution"] += 1
                print(std_dic)
                print(std_eye)

        cv2.putText(test, str(std_cnt["eye_caution"]), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (128,255,255),3)
        cv2.imshow(str(self.s_number) + 'janus', test)
        cv2.waitKey(1) & 0xFF
        return frame

class JanusPlugin:
    def __init__(self, session, url):
        self._queue = asyncio.Queue()
        self._session = session
        self._url = url

    async def send(self, payload):
        message = {"janus": "message", "transaction": transaction_id()}
        message.update(payload)
        async with self._session._http.post(self._url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "ack"

        response = await self._queue.get()
        assert response["transaction"] == message["transaction"]
        return response


class JanusSession:
    def __init__(self, url):
        self._http = None
        self._poll_task = None
        self._plugins = {}
        self._root_url = url
        self._session_url = None

    async def attach(self, plugin_name: str) -> JanusPlugin:
        message = {
            "janus": "attach",
            "plugin": plugin_name,
            "transaction": transaction_id(),
        }
        async with self._http.post(self._session_url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "success"
            plugin_id = data["data"]["id"]
            plugin = JanusPlugin(self, self._session_url + "/" + str(plugin_id))
            self._plugins[plugin_id] = plugin
            return plugin

    async def create(self):
        self._http = aiohttp.ClientSession()
        message = {"janus": "create", "transaction": transaction_id()}
        async with self._http.post(self._root_url, json=message) as response:
            data = await response.json()
            assert data["janus"] == "success"
            session_id = data["data"]["id"]
            self._session_url = self._root_url + "/" + str(session_id)

        self._poll_task = asyncio.ensure_future(self._poll())

    async def destroy(self):
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None

        if self._session_url:
            message = {"janus": "destroy", "transaction": transaction_id()}
            async with self._http.post(self._session_url, json=message) as response:
                data = await response.json()
                assert data["janus"] == "success"
            self._session_url = None

        if self._http:
            await self._http.close()
            self._http = None

    async def _poll(self):
        while True:
            params = {"maxev": 1, "rid": int(time.time() * 1000)}
            async with self._http.get(self._session_url, params=params) as response:
                data = await response.json()
                if data["janus"] == "event":
                    plugin = self._plugins.get(data["sender"], None)
                    if plugin:
                        await plugin._queue.put(data)
                    else:
                        print(data)

async def subscribe(session, room, feed, s_number, test_id):
    pc = RTCPeerConnection()
    pcs.add(pc)
    s_num = s_number
    @pc.on("track")
    async def on_track(track):
        print("Track %s received" % track.kind)
        if track.kind == "video":
            while True:
                await VideoTransformTrack(track, s_num, test_id).recv()

    # subscribe
    plugin = await session.attach("janus.plugin.videoroom")
    response = await plugin.send(
        {"body": {"request": "join", "ptype": "subscriber", "room": room, "feed": feed}}
    )

    #print(response)
    # apply offer
    await pc.setRemoteDescription(

        RTCSessionDescription(
            sdp=response["jsep"]["sdp"], type=response["jsep"]["type"]
        )
    )

    # send answer
    await pc.setLocalDescription(await pc.createAnswer())
    response = await plugin.send(
        {
            "body": {"request": "start"},
            "jsep": {
                "sdp": pc.localDescription.sdp,
                "trickle": False,
                "type": pc.localDescription.type,
            },
        }
    )

async def run(player, recorder, room, session, test_id, s_number):

    await session.create()

    # join video room
    plugin = await session.attach("janus.plugin.videoroom")
    response = await plugin.send(
        {
            "body": {
                "display": "aiortc",
                "ptype": "publisher",
                "request": "join",
                "room": room,
            }
        }
    )

    publishers = response["plugindata"]["data"]["publishers"]

    for publisher in publishers:
        print("id: %(id)s, display: %(display)s" % publisher)

    maxlength = len(publishers)
    print(s_number, ' stream')
    test_num = test_id
    # receive video
    if maxlength is s_number:
        print('no stream on janus')
    else:
        for index in range(0, maxlength):
            std_id = int(publishers[index]['display'])
            if std_id == "null":
                pass
            else:
                if std_id % s_number is 0:
                    std_dic[std_id] = {
                        "test_id" : test_id,
                        "s_number": std_id,
                    }
                    std_eye[std_id] = {
                        "s_number" : std_id,
                        "eye_caution": 0
                    }
                    print(std_dic)
                    print(std_eye)
                    await subscribe(
                        session=session, room=room, feed=publishers[index]["id"], s_number=std_id, test_id=test_num
                    )

def janus_connection(url, room, test_id, s_number):

    session = JanusSession(url)
    #state()
    face()
    player = None
    recorder = None
    #asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()

    task = loop.run_until_complete(
        run(player=player, recorder=recorder, room=room, session=session, test_id=test_id, s_number=s_number)
    )
    loop.run_forever()


#janus_connection("https://re-coder.net/janus", 1234, 50, 21)