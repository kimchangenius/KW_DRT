import enum


class ActionType(enum.Enum):
    REJECT = 0
    PICKUP = 1
    DROPOFF = 2

    def __str__(self):
        return self.name
