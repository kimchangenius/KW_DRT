from enum import IntEnum

VEHICLE_STATUS_NUM_CLASSES = 4

class VehicleStatus(IntEnum):
    IDLE = 1
    REJECT = 2
    PICKUP = 3
    DROPOFF = 4

    DUMMY = 0

    def __str__(self):
        return self.name
