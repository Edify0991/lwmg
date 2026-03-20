from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lwmg.sonic_io.sonic_zmq_publisher import SonicZmqPublisher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    if cfg["zmq"].get("enabled", False):
        pub = SonicZmqPublisher(bind=cfg["zmq"]["bind"])
        pub.publish_qpos([0.0] * 12)
        print("published one qpos sample")
    else:
        print("zmq disabled; mock sim2sim no-op")


if __name__ == "__main__":
    main()
