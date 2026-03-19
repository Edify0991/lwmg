from __future__ import annotations

import json
from dataclasses import dataclass

import zmq


@dataclass
class SonicZmqPublisher:
    bind: str = "tcp://127.0.0.1:5555"

    def __post_init__(self) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(self.bind)

    def publish_qpos(self, qpos: list[float]) -> None:
        self.socket.send_string(json.dumps({"qpos": qpos}))
