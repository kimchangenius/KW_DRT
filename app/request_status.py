from enum import IntEnum

REQUEST_STATUS_NUM_CLASSES = 3

class RequestStatus(IntEnum):
    PENDING = 1     # 어떤 차량도 수락하지 않은 상태
    ACCEPTED = 2    # 차량 한 대에 의해 수락된 사태
    PICKEDUP = 3    # 차량이 승객을 태운 상태, drop_off action 후보 가능 상태

    # For Information Recording
    INVALID = 0  # 빈 요청 슬롯 or 완료된 요청
    CANCELLED = 4
    SERVED = 5

    def __str__(self):
        return self.name
