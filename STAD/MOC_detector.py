'''修改后处理get_result函数，使其通过参考信息返回匹配后的结果'''
import torch
import numpy as np
import cv2
from collections import deque

from .utils import iou2d, nms_tubelets,giou,bbox_edge_distance
from .MOC_utils.data_parallel import DataParallel
from .MOC_utils.decode import moc_decode
from .MOC_utils.network.moc_det import MOC_Det, MOC_Backbone
from torchvision.ops import nms

np.seterr(divide='ignore',invalid='ignore')

def apply_nms(bbox_list, iou_threshold=0.5):
    """
    对bbox_list进行NMS操作。
    
    参数：
    - bbox_list: 包含多个五维bbox的list。每个bbox的形状为 (H, W, 4, confidence)，
                 其中4表示bbox的坐标 (x1, y1, x2, y2)，confidence表示置信度。
    - iou_threshold: NMS操作的IOU阈值，默认值为0.5。

    返回：
    - 返回经过NMS处理后的bbox，形状与原数据一致。
    """
    result = []
    for bbox in bbox_list:
        # 假设bbox的形状为(H, W, 4, 1)，在这里提取xyxy坐标和置信度
        coordinates = bbox[..., :4]  # 获取坐标 (x1, y1, x2, y2)
        scores = bbox[..., 4]  # 获取置信度

        # 将bbox和置信度平展为一维
        coordinates = coordinates.reshape(-1, 4)
        scores = scores.reshape(-1)

        # 将坐标和置信度转换为tensor
        coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # 使用nms
        keep = nms(coordinates_tensor, scores_tensor, iou_threshold)

        # 过滤掉非保留的框
        result_bboxes = coordinates_tensor[keep].cpu().numpy()
        result.append(result_bboxes)
    
    return result

def create_inference_model(arch, branch_info, head_conv, K, flip_test=False):
    # 构建模型
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    backbone = MOC_Backbone(arch, num_layers)
    branch = MOC_Det(backbone, branch_info, arch, head_conv, K, flip_test=flip_test)
    return backbone, branch

def load_inference_model(backbone, branch, model_path):
    # 加载模型
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    backbone.load_state_dict(state_dict, strict=False)

    branch.load_state_dict(state_dict, strict=False)

    return backbone, branch

# calculate average scores of tubelets in tubes
def tubescore(tt):
    return np.max(np.array([tt[i][1][-1] for i in range(len(tt))]))

class MOCDetector(object):
    def __init__(self, configs):
        # 默认使用单卡
        gpus = [0]

        '''MOC模型结构相关参数'''
        # arch根据选择的backbone确定，使用的是dla_34或者resnet_18
        arch = configs['backbone']
        branch_info = {'hm': 1, 'hm_2': 2,'mov': 14, 'wh': 14}
        head_conv=256

        self.configs = configs
        ''' MOC模型推理相关参数 '''
        # 初始化帧队列长度
        self.K = configs['K']

        # 模型路径
        rgb_model = configs['model_path']

        # 初始化帧队列,frame_queue为 7 帧队列，每7张图片进行一次推理；
        self.frame_queue = deque(maxlen=self.K)

        # 类别数，目前只有一类为：投球
        self.num_classes = self.configs['num_classes']

        # 当前传入的帧数
        self.i = 0

        # 临时存储MOC推理结果，通过get_result函数获得最终结果
        self.or_tubes = {}

        # 导入模型初始化
        if gpus[0] >= 0:
            device = torch.device('cuda')
        else:
            assert 'cpu is not supported!'

        self.rgb_model_backbone, self.rgb_model_branch = None, None
        self.flow_model_backbone, self.flow_model_branch = None, None

        self.rgb_model_backbone, self.rgb_model_branch = create_inference_model(arch, branch_info, head_conv, self.K, flip_test=False)
        print('create rgb model', flush=True)
        self.rgb_model_backbone, self.rgb_model_branch = load_inference_model(self.rgb_model_backbone, self.rgb_model_branch, rgb_model)
        print('load rgb model', flush=True)
        self.rgb_model_backbone = DataParallel(
            self.rgb_model_backbone, device_ids=[gpus[0]],
            chunk_sizes=[1]).to(device)
        self.rgb_model_branch = DataParallel(
            self.rgb_model_branch, device_ids=[gpus[0]],
            chunk_sizes=[1]).to(device)
        print('put rgb model to gpu', flush=True)
        self.rgb_model_backbone.eval()
        self.rgb_model_branch.eval()

        self.rgb_buffer = []
        self.rgb_buffer_flip = []

    def preprocess(self, frame, resize_width, resize_height):
        # 预处理函数
        """
            对K帧输入帧进行预处理
            :param frame: 帧
            :param resize_width: 推理的图片宽度
            :param resize_height: 推理的图片高度
            :return data: 预处理后的数据
        """
        # 设定归一化的均值和方差
        or_mean = [0.40789654, 0.44719302, 0.47026115]
        or_std = [0.28863828, 0.27408164, 0.27809835]

        #对队列中的图像进行预处理
        # 需要优化
        images = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

        data = np.empty((3, resize_height, resize_width), dtype=np.float32)

        mean = np.tile(np.array(or_mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        std = np.tile(np.array(or_std, dtype=np.float32)[:, None, None], (1, 1, 1))
        
        data[0: 3, :, :] = np.transpose(images, (2, 0, 1))

        # normalize
        data = ((data / 255.) - mean) / std

        return data

    def post_process(self, detections, height, width, output_height, output_width, num_classes, K):
        # 后处理函数
        """
            对K帧输入帧进行预处理
            :param detections: 检测结果(模型输出)
            :param height: 原图的高度
            :param width: 原图的宽度
            :param output_height: 模型输出的高度(特征图高度)
            :param output_width: 模型输出的宽度(特征图宽度)
            :param num_classes: 类别数
            :param K: 推理的帧队列的长度
            :return: data: 后处理后的数据
        """
        # 数据转换，cpu上进行后处理
        detections = detections.detach().cpu().numpy()

        results = []
        for i in range(detections.shape[0]):
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * K + 1].astype(np.float32)
            results.append(top_preds)
        return results

    def process(self, images, K):
        # 模型推理函数
        """
        ：param images: K帧输入帧
        ：param K: 帧队列长度
        ：return: detections: 检测结果
        """
        with torch.no_grad():
            if self.rgb_model_backbone is not None:
                if len(self.rgb_buffer) < K:
                    rgb_features = [self.rgb_model_backbone(torch.tensor(images[i][None])) for i in range(K)]
                    self.rgb_buffer = rgb_features
                elif len(self.rgb_buffer) == K:
                        del self.rgb_buffer[0]
                        self.rgb_buffer.append(self.rgb_model_backbone(torch.tensor(images[K-1][None])))

                rgb_output = self.rgb_model_branch(self.rgb_buffer, self.rgb_buffer_flip)
                rgb_hm = rgb_output[0]['hm_2'].sigmoid_()
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']

                hm = rgb_hm
                wh = rgb_wh
                mov = rgb_mov
            detections = moc_decode(hm, wh, mov, N=10, K=K)
            return detections

    def moc_inference(self, frame, frame_index):
        # 模型推理主函数
        """
        ：param frame: 输入帧
        ：param frame_index: 输入的帧index
        ：return: or_tubes: 检测结果
        """

        if self.i == 0:
            self.start_frame = frame_index

        self.i += 1

        self.step = self.configs['step']

        # 获取原始图片的宽度和高度
        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])

        # 对输入的单帧图像进行预处理，再存入帧队列准备推理
        pre_frame = self.preprocess(frame, self.configs['resize_width'], self.configs['resize_height'])
        self.frame_queue.append(pre_frame)

        # 当帧队列长度等于K时，进行推理
        if len(self.frame_queue) == self.K:

            # 对预处理后的数据进行推理
            detections = self.process(self.frame_queue, self.K)

            # 对检测结果进行后处理
            results = self.post_process(detections, frame_height, frame_width, output_height=self.configs['resize_height']/4, output_width=self.configs['resize_width']/4, num_classes=self.num_classes, K=self.K)

            # 将推理结果存入or_tubes
            self.or_tubes[frame_index-6*self.step] = results[0]

            return self.or_tubes

    def get_result(self, ref_info):
        # 获取最终结果，在对一个小的视频片段进行推理之后，调用该函数获取最终结果
        """
        ：param ref_info: 参考信息   {'shot_index': frame_in, 'ball_bbox': [1204, 558, 35, 37]}
        ：return: action_class: 动作类别
        ：return: bboxes: 每帧上的位置（边界框）信息，格式为字典，键为帧号，值为bbox
        ：return: key_result: 关键帧结果，键为帧号，值为bbox
        """
        RES = {}
        for ilabel in range(self.num_classes):
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)

            for frame in range(self.start_frame, self.start_frame+len(self.or_tubes)*self.step, self.step):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                ltubelets = self.or_tubes[frame][ilabel + 1]  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score

                # nms_tubelets对每个类别的tubelets进行nms，返回的是nms后的tubelets
                ltubelets = nms_tubelets(ltubelets, self.configs['nms_tubelets_threshold'], top_k=self.configs['top_k'])

                # just start new tubes
                if frame == self.start_frame:
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(self.start_frame, ltubelets[i, :])])
                    continue

                # sort current tubes according to max score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    offset = frame - last_frame
                    if offset < self.K*self.step:
                        nov = self.K - offset
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4], last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in range(nov)]) / float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * self.K - 4:4 * self.K])

                    valid = np.where(ious >= 0.5)[0]

                    if valid.size > 0:
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= self.K*self.step:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)
                # print(score)

                # just start new tubes
                if score < self.configs['tube_score_threshold']:
                    continue

                # print(score)
                beginframe = t[0][0]
                endframe = t[-1][0]
                length = int((endframe + self.step - beginframe)/self.step + self.K-1)
               
                # delete tubes with short duraton
                if length < self.configs['tubes_length_threshold']:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + self.K*self.step, step=self.step)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(self.K):
                        out[int((frame - beginframe)/self.step) + k, 1:5] += box[4 * k:4 * k + 4]
                        out[int((frame - beginframe)/self.step) + k, -1] += box[-1]  # single frame confidence
                        n_per_frame[int((frame - beginframe)/self.step) + k, 0] += 1
                out[:, 1:] /= n_per_frame
                output.append([out, score])

            RES[ilabel] = output
        
        # 如果moc没有检测结果，返回None
        if len(RES[0]) == 0:
            return None, None, None

        # 如果有有效的参考信息，则进行筛选
        if ref_info['ball_bbox'] is not None:
            action_class, ture_tubes, key_results = self.use_ref_info(ref_info, RES)
        else:
            action_class, ture_tubes, key_results = self.get_max_confidence_result(RES)

        action_class, ture_tubes, key_results = self.get_all_result_with_nms(RES)

        # # 修剪最终结果，去掉置信度很低的框、保留以关键帧为中心前后的框
        # # # 单类别的处理方式
        # delete_keys = []
        # for tube in ture_tubes:
        #     bbox = ture_tubes[tube]
        #     if (abs(tube-key_frame_index) > 8*self.configs['step'] and bbox[-1] < 0.15) or abs(tube-key_frame_index) > 12*self.configs['step']:
        #         # 如果里关键帧帧号大于10帧且置信度小于-0.15，则删除
        #         delete_keys.append(tube)
        # for key in delete_keys:
        #     del ture_tubes[key]



        return action_class, ture_tubes, key_results
        
    def use_ref_info(self, ref_info, RES):
        # use_ref_info函数，如果有参考信息ref_info，则调用该函数对结果进行筛选
        """
        ：param ref_info: 参考信息 
        ：param RES: MOC关联后的检测结果
        ：return action_class: 动作类别 ture_tubes:tube的信息 key_results:关键帧结果
        """
        action_class, bboxes, key_result = 0, {}, {}
        key_frame_bbox = None
        # 获取参考信息,shot_index为关键帧帧号，ball_bbox为篮球位置信息
        shot_index = ref_info['shot_index']
        ball_bbox = ref_info['ball_bbox']
        ball_bbox = [ball_bbox[0], ball_bbox[1], ball_bbox[0] + ball_bbox[2], ball_bbox[1] + ball_bbox[3]] # x1,y1,x2,y2

        # 获取MOC各个tube的关键帧号与关键帧球员信息
        closest_frame_player = {} # 用于存储距离参考帧最近帧的球员位置信息，字典的key为帧号，value为对应帧号的球员位置信息
        key_frame_confid = {} # 用于存储距离参考帧最近帧的球员位置信息的置信度，字典的key为帧号，value为对应帧号的球员位置信息的置信度
        key_frame_player = {} # 用于存储关键帧的球员位置信息，字典的key为帧号，value为对应帧号的球员位置信息
        player_tube = {} # 用于存储球员位置信息与帧号的对应关系，字典的key为帧号，value为对应帧号的球员位置信息
        for action_class in RES:
            for tube in RES[action_class]:
                # 根据置信度对bbox进行排序
                sorted_bboxs = sorted(tube[0], key=lambda x: x[5], reverse=True)
                # 保留置信度最高的bbox，即为key_frame的results
                highest_confi_box = sorted_bboxs[0]
                # shot_index最接近的player_bbox的球员位置信息
                dists = []
                for bbox in tube[0]:
                    dists.append(abs(bbox[0] - shot_index))
                min_dist = min(dists)
                index = dists.index(min_dist)
                closest_frame_player[highest_confi_box[0]] = list(tube[0][index][1:5])
                key_frame_confid[highest_confi_box[0]] = highest_confi_box[5]

                # key_frame的player_bbox的球员位置信息
                key_frame_player[highest_confi_box[0]] = highest_confi_box[1:5]
                Tube = {}
                for box in tube[0] :
                    Tube[box[0]] = box[1:5]
                player_tube[highest_confi_box[0]] = Tube

        # 主要以空间信息作为基准，筛选出与篮球位置信息相近的tube
        dist_to_ball = {} # 用于存储与篮球位置信息相近的player_bbox,字典的key为player_bbox，value为与篮球位置信息的距离
        
        # 使用giou进行筛选
        # for index, player in enumerate(key_player_bbox):
        #     if giou(player, ball_bbox) > -0.4:
        #         index_dic[index] = giou(player, ball_bbox)

        # 使用bbox_edge_distance进行筛选
        for frame_id, player_bbox in closest_frame_player.items():
            if bbox_edge_distance(ball_bbox, player_bbox) < self.configs['max_dist']:
                dist_to_ball[frame_id] = bbox_edge_distance(player_bbox, ball_bbox)

        # 设置底线，如果frame_id相差太大，则认为没对应上
        keys_to_delete = []
        for frame_id, dist in dist_to_ball.items():
            if abs(frame_id - shot_index) > self.configs['max_frame_gap'] * 30 * self.configs['step']:
                keys_to_delete.append(frame_id)
        keys_to_delete = list(set(keys_to_delete))
        for key in keys_to_delete:
            del dist_to_ball[key]

        score_to_ball = {}
        # 当有dist_to_ball中有多个结果，设计一个函数，分数最小的为最终结果
        # 函数同时考虑三个方面的问题
        # 1. key_frame是否小于shot_index
        # 2. giou的值
        # 3. frame_id最接近
        if len(dist_to_ball) > 1:
            keys_to_delete = []
            for frame_id, dist in dist_to_ball.items():
                # 考虑key_frame是否小于shot_index
                if frame_id < shot_index:
                    weight = 1.3
                else:
                    weight = 0.7
                # 考虑frame_id最接近
                frame_gap = abs(frame_id - shot_index) / (80 * self.configs['step'])
                score_to_ball[frame_id] = (dist + frame_gap) * weight
            # 只保留value最小的
            min_value = min(score_to_ball.values())
            score_to_ball = {key: value for key, value in score_to_ball.items() if value == min_value}
            # 解析结果
            for key_frame, _ in score_to_ball.items():
                key_frame_bbox = key_frame_player[key_frame]
                bboxes = player_tube[key_frame]

        # 如果dist_to_ball只有一个结果，直接进行解析
        if len(dist_to_ball) == 1: 
            for key_frame, _ in dist_to_ball.items():
                key_frame_bbox = key_frame_player[key_frame]
                bboxes = player_tube[key_frame]

        # # # 如果dist_to_ball没有结果(参考的出手帧离MOC的结果都很远),
        # # 方案1
        # # 则返回keyframe与shot_index最接近的结果
        # if len(dist_to_ball) == 0:
        #     time_gap = {}
        #     for key_frame, _ in dist_to_ball.items():
        #         time_gap[key_frame] = (abs(key_frame - shot_index))
        #     min_time_gap = min(time_gap)
        #     time_gap = {key: value for key, value in time_gap.items() if value == min_time_gap}
        #     for key_frame, _ in time_gap.items():
        #         key_frame_bbox = key_frame_player[key_frame]
        #         bboxes = player_tube[key_frame]

        # 如果dist_to_ball没有结果(参考的出手帧离MOC的结果都很远)
        # 方案2
        # 则返回最大置信度的结果
        if len(dist_to_ball) == 0:
            action_class, bboxes, key_frame, key_frame_bbox = self.get_max_confidence_result(RES)

        # # 如果dist_to_ball没有结果(参考的出手帧离MOC的结果都很远)，
        # 方案3
        # 同时考虑时间差和置信度
        # if len(dist_to_ball) == 0:
        #     time_gap = {}
        #     for key_frame, _ in dist_to_ball.items():
        #         time_gap[key_frame] = (abs(key_frame - shot_index))

        #     # 计算分数
        #     max_time_gap = max(list(time_gap.values()))
        #     max_confid = max(list(key_frame_confid.values()))
        #     score = {}
        #     for key_frame, _ in time_gap.items():
        #         score[key_frame] = time_gap[key_frame]/max_time_gap + 1/(key_frame_confid[key_frame]/max_confid)

        #     min_score = min(score.values())
        #     score = {key: value for key, value in score.items() if value == min_score}
        #     for key_frame, _ in score.items():
        #         key_frame_bbox = key_frame_player[key_frame]
        #         bboxes = player_tube[key_frame]   

        return action_class, bboxes, key_frame, key_frame_bbox


    def get_max_confidence_result(self, RES):
        action_classes, tubes, key_frame_bboxes = [], [], []
        # 解析结果，与接口输出格式对应
        for action_class in RES:
            bboxes = {}
            key_result = {}
            key_frame_bbox = {}
            # 根据置信度对tube进行排序
            sorted_tubes = sorted(RES[action_class], key=lambda x: x[1], reverse=True)
            # 保留置信度最高的tube
            if len(sorted_tubes) == 0:
                continue
            highest_confi_tube = sorted_tubes[0]

            # 根据置信度对bbox进行排序
            sorted_bboxs = sorted(highest_confi_tube[0], key=lambda x: x[5], reverse=True)
            # 保留置信度最高的bbox
            highest_confi_box = sorted_bboxs[0]

            # 解析结果为字典格式key_result，bboxes
            # key_result['frame_id'] = int(highest_confi_box[0])
            # key_result['bbox'] = highest_confi_box[1:5].astype(int).tolist()
            key_result[highest_confi_box[0]] = highest_confi_box[1:5]
            key_frame_index = int(highest_confi_box[0])
            key_frame_bbox[key_frame_index] = highest_confi_box[1:5]

            for key in highest_confi_tube[0]:
                # bboxes[int(key[0])] = key[1:5].astype(int).tolist()
                bboxes[key[0]] = key[1:6]

            action_classes.append(action_class)
            tubes.append(bboxes)
            key_frame_bboxes.append(key_frame_bbox)

        return action_classes, tubes, key_frame_bboxes

    def get_all_result_with_nms(self, RES):
        action_classes, tubes, key_frame_bboxes = [], [], []
        # 解析结果，与接口输出格式对应
        for action_class in RES:      
            # 对每一个tube进行处理
            for tube in RES[action_class]:
                bboxes = {}
                key_result = {}
                key_frame_bbox = {}
                # 根据置信度对bbox进行排序
                sorted_bboxs = sorted(tube[0], key=lambda x: x[5], reverse=True)
                # 保留置信度最高的bbox
                highest_confi_box = sorted_bboxs[0]

                # 解析结果为字典格式key_result，bboxes
                # key_result['frame_id'] = int(highest_confi_box[0])
                # key_result['bbox'] = highest_confi_box[1:5].astype(int).tolist()
                key_result[highest_confi_box[0]] = highest_confi_box[1:5]
                key_frame_index = int(highest_confi_box[0])
                key_frame_bbox[key_frame_index] = highest_confi_box[1:5]

                for key in tube[0]:
                    # bboxes[int(key[0])] = key[1:5].astype(int).tolist()
                    bboxes[key[0]] = key[1:6]

                action_classes.append(action_class)
                tubes.append(bboxes)
                key_frame_bboxes.append(key_frame_bbox)

        # 获取最长的tube

        return action_classes, tubes, key_frame_bboxes

    def reset(self):
        # 重置模型参数
        """
        reset模型参数
        主要清空self.frame_queue和self.or_tubes
        将self.i置为0
        """
        self.frame_queue = deque(maxlen=self.K)
        self.i = 0
        self.or_tubes = {}
        self.rgb_buffer = []
        self.rgb_buffer_flip = []

