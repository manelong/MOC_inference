from enum import Enum

class EventType(Enum):
    """
    篮球事件类型，0：无事件，301：投篮，302：篮板，303：抢断，304：盖帽
    """
    NO_EVENT = 0
    SHOT = 301
    REBOUND = 302
    STEAL = 303
    BLOCK = 304
    SHOT_START = 305  # 投球出手时刻事件名称

class SingleAction:
    def __init__(self, event_type, bboxes, key_result):
        """
        初始化一个事件

        :param event_type: 事件类别
        :param bboxes: 每帧上的位置（边界框）信息，格式为字典，键为帧号，值为bbox
        :param key_result: 关键帧结果，键为帧号，值为bbox
        """
        self.event_type = event_type
        self.bboxes = bboxes
        self.key_result = key_result