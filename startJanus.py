import getJanusTracking
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import aiohttp
import socketio
import threading
from multiprocessing import Process

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder




def callJanus(url, room, test_id, s_number):
    #getJanus.janus_connection(url, room, test_id, s_number)
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=getJanus.janus_connection, args=(url, room, test_id, s_number + 0, loop)).start()
    loop = asyncio.new_event_loop()
    threading.Thread(target=getJanus.janus_connection, args=(url, room, test_id, s_number + 1, loop)).start()
    """
    getJanusTracking.janus_connection(url, room, test_id, s_number)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Janus")
    parser.add_argument("url", help="Janus URL", default="https://re-coder.net/janus")
    parser.add_argument("room", type=int, help="room ID to join", default=1234)
    parser.add_argument("test_id", type=int, help="test_id", default=50)
    parser.add_argument("s_number", type=int, help="s_number", default=21)
    args = parser.parse_args()


    callJanus(url="https://re-coder.net/janus", room=args.room, test_id=args.test_id, s_number=args.s_number)
