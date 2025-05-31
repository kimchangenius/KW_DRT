from enum import IntEnum


class VehicleStatus(IntEnum):
    DUMMY = 0
    IDLE = 1
    REJECT = 2
    PICKUP = 3
    DROPOFF = 4

    NUM_CLASSES = 4
