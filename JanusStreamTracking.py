import subprocess
import argparse
import asyncio
import logging
import random
import string
import time
import aiohttp

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

pcs = set()

std_numlist = []
std_count = 0

def transaction_id():
    return "".join(random.choice(string.ascii_letters) for x in range(12))

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

# janus server에 접속해서 학생들의 수 얻어오기
async def run(room, session, test_id):
    global std_count

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
        if int(publisher['display']) % 3 is 0:
            std_numlist.append(int(publisher['display']))

    std_count =  len(std_numlist)

# 서브 프로세스 생성 함수
def openJanus(url, room, test_id, id):
    proc = subprocess.Popen(['python3', 'startJanus.py', str(url), str(room), str(test_id), id])
    return proc

# 웹캠 스트림이 연결 된 학생 수 만큼 프로세스 생성
def runJanus(url, room, test_id):
    global std_count, std_numlist

    process = []
    count = 0

    while True:
        if count >= std_count:
            break

        # 야누스 스트림에 접속하기 위한 인자 값 전달, 서브 프로세스 생성
        janus_process = openJanus(url=url, room=room, test_id=test_id, id=str(std_numlist[count]))
        process.append(janus_process)
        count += 1

    # 열어둔 프로세스 동시 실행
    for proc in process:
        proc.communicate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janus")
    parser.add_argument("url", help="Janus URL", default="https://re-coder.net/janus")
    parser.add_argument("room", type=int, help="room ID to join", default=1234)
    parser.add_argument("test_id", type=int, help="test_id", default=50)

    args = parser.parse_args()

    session = JanusSession(args.url)

    loop = asyncio.get_event_loop()

    # janus server에 접속한 학생 수 얻어오기
    loop.run_until_complete(
        run(room=args.room, session=session, test_id=args.test_id)
    )

    # janus 서버 연결 및, 스트림 데이터로 아이트래킹 시작
    runJanus(url=args.url, room=args.room, test_id=args.test_id)
