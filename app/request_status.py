from enum import IntEnum


class RequestStatus(IntEnum):
    INVALID = 0     # 빈 요청 슬롯 or 완료된 요청
    PENDING = 1     # 어떤 차량도 수락하지 않은 상태
    ACCEPTED = 2    # 차량 한 대에 의해 수락된 사태
    PICKEDUP = 3    # 차량이 승객을 태운 상태, drop_off action 후보 가능 상태

    NUM_CLASSES = 3
