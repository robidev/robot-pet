#!/usr/bin/env python3
"""
Read-only status dump. Safe to run at any time -- never moves the robot.

Usage: python3 status.py [ip-address]  (defaults to 192.168.101.43)
"""

import sys

from valetudo_client import ValetudoClient


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "192.168.101.43"
    vac = ValetudoClient(host)

    info = vac.get_robot_info()
    print(f"{info['manufacturer']} {info['modelName']} ({info['implementation']})")
    print(f"capabilities: {', '.join(vac.get_capabilities())}")

    status = vac.get_status()
    battery = vac.get_battery()
    pose = vac.get_position()
    charger = vac.get_charger_position()

    print(f"status:  {status['value']} (flag={status['flag']})")
    if status["error"]:
        print(f"error:   {status['error']}")
    print(f"battery: {battery['level']}% ({battery['flag']})")
    if pose:
        print(f"pose:    x={pose.x} y={pose.y} angle={pose.angle}")
    if charger:
        print(f"charger: x={charger.x} y={charger.y}")


if __name__ == "__main__":
    main()
