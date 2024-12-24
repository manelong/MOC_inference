# 用于对篮球视频进行动作分析，包括投篮、传球等动作的检测
from STAD import MOC_detector
from action_data_type import SingleAction

class ActionAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger        #日志，一些需要捕获的error信息需要使用
        self.stad_analyer = None    #指向stad算法分析器

    def initialize(self, configs):
        """
        初始化模型
        :param configs: dict，模型配置，包括路径以及一些必要配置参数
        :return:
        """
        self.configs = configs
        # 初始化模型
        self.STAD_detector = MOC_detector.MOCDetector(self.configs)

        return True

    def process(self, frame_rgb, frame_index):
        """
        对输入帧进行检测
        :param frame_rgb:   输入帧
        :param frame_index: 当前帧数

        """
        self.STAD_detector.moc_inference(frame_rgb, frame_index)

    def get_results(self, ref_info=None):
        """
        获取结果
        :return: list of SingleAction
        """
        action_class, bboxes, key_result = self.STAD_detector.get_result(ref_info)
        # print(f"action_class: {action_class}, bboxes: {bboxes}, key_result: {key_result}")

        return SingleAction(action_class, bboxes, key_result)

    def reset(self):
        """
        重置模型参数
        """
        self.STAD_detector.reset()
        
