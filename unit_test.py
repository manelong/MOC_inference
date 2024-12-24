'''用于测试多个视频片段，一般是用来进行测试集、验证集的可视化，以及个别case的重演
对于单个clip视频段，整体作后处理，不进行推理时间范围的划定'''

'''修改后处理'''
from action_analyzer import ActionAnalyzer
import cv2
import os
import json

def vis(input_file, output_file_path, result, config, ref_info ,ball_trace):
    '''
    param input_file: 视频文件路径
    param result: 检测结果

    return: None
    当前目录下生成结果视频文件
    '''
    assert os.path.isfile(input_file), f"{input_file} not exists"
    cap = cv2.VideoCapture(input_file)
    output_path = os.path.join(output_file_path, os.path.basename(input_file))

    # 获取视频的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个VideoWriter对象
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame_rgb = cap.read()
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 如果没有读取到帧，则退出循环
        if not ret:
            break
        
        if frame_id % config['step'] == 0:
            if result.bboxes != None:
                if frame_id in list(result.bboxes.keys()) :
                    x1, y1, x2, y2, confid = result.bboxes[frame_id]
                    print(frame_id)
                    cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame_rgb, f'{confid:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(confid)

            # if frame_id in list(ball_trace.keys()):
            #     x1, y1, x2, y2 = ball_trace[frame_id]
            #     cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x1)+int(x2), int(y1)+int(y2)), (0, 255, 0), 2)
            
            cv2.putText(frame_rgb, f'{frame_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
            out.write(frame_rgb)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('done')

def vis_2(input_file, output_file_path, result, config, ref_info ,ball_trace):
    '''
    param input_file: 视频文件路径
    param result: 检测结果

    return: None
    当前目录下生成结果视频文件
    '''
    assert os.path.isfile(input_file), f"{input_file} not exists"
    cap = cv2.VideoCapture(input_file)
    output_path = os.path.join(output_file_path, os.path.basename(input_file))

    # 获取视频的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个VideoWriter对象
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame_rgb = cap.read()
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 如果没有读取到帧，则退出循环
        if not ret:
            break
        
        # 对不同的动作类型结果同时进行可视化

        if frame_id % config['step'] == 0:
            if result.bboxes != None:
                for idx, res in enumerate(result.bboxes):
                    if frame_id in list(res.keys()) :
                        x1, y1, x2, y2, confid = res[frame_id]
                        print(frame_id)
                        if confid < 0.2:
                            continue
                        cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        if result.event_type[idx] == 0:
                            cv2.putText(frame_rgb, f'{confid:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame_rgb, f'{result.event_type[idx]}', (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif result.event_type[idx] == 1:
                            cv2.putText(frame_rgb, f'{confid:.2f}', (int(x1), int(y1-60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame_rgb, f'{result.event_type[idx]}', (int(x1), int(y1)-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(confid)

            # if frame_id in list(ball_trace.keys()):
            #     x1, y1, x2, y2 = ball_trace[frame_id]
            #     cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x1)+int(x2), int(y1)+int(y2)), (0, 255, 0), 2)
            
            cv2.putText(frame_rgb, f'{frame_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
            out.write(frame_rgb)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('done')

def get_video_path(video_folder):
    # 获取文件夹下所有视频文件
    video_files = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files

def get_ref_info(json_file):
    # 从一个json文件中读取到参考信息，返回一个字典
    # {shot_index:帧号, ball_bbox:位置信息}
    with open(json_file, 'r') as f:
        ai_json = json.load(f)
    ai_datas = ai_json['frame_list']
    fps_ratio = ai_json['fps_ratio']
    print(f"fps_ratio:{fps_ratio}")

    key_frame_index = 0
    key_frame_bbox = None
    ball_trace = None
    new_ball_trace = {}
    ref_info = {'shot_index': None, 'ball_bbox': key_frame_bbox}
    for i, data in enumerate(ai_datas):
        if data['event'] == None:
            continue
        if data['event']['event_type'] == 305:
            key_frame_index = i * fps_ratio
            print(f"i: {i}, key_frame_index:{key_frame_index}, frame_index:{data['frame_index']}")
            ball_trace = data['event']['event_info']['ball_trace']
            if ball_trace is not None:
                key_frame_bbox = ball_trace[0]
                for j, ball in enumerate(ball_trace):
                    new_ball_trace[key_frame_index] = [int(ball[0]), int(ball[1]), int(ball[2]), int(ball[3])]
                    key_frame_index += fps_ratio

                ref_info = {'shot_index': list(new_ball_trace.keys())[0], 'ball_bbox': key_frame_bbox}
            else:
                ref_info = {'shot_index': None, 'ball_bbox': key_frame_bbox}
    return ref_info, new_ball_trace

if __name__ == '__main__':

    config = {
        'model_path': 'pretrained\model_[35]_new_center',  # 模型路径
        'resize_width': 512,  # 图片resize宽度
        'resize_height': 288,  # 图片resize高度
        'K': 7,  # 帧队列长度
        'num_classes': 2,  # 类别数
        'step': 1,  # 帧间隔
        'nms_tubelets_threshold': 0.4,  # stubelets的nms阈值
        'top_k': 10,  # nms过滤保留top_k的tube
        'tubes_length_threshold': 5,  # tube的长度阈值,小于这个阈值的tube进行删除
        'tube_score_threshold': 0.4,  # tube的分数阈值,小于这个阈值的tube进行删除
        'backbone': 'yolo',# 网络的backbone,现在可选dla_34,resnet_18,yolo
        'max_dist':3.5, # 篮球出手bbox置与投球运动员bbox允许最大距离，例：3.5表示3.5倍篮球bbox宽度
        'max_frame_gap': 1.5, # 篮球出手bbox与投球运动员bbox允许最大时间差，例：1.5表示1.5s
    }

    # 初始化
    action_analyzer = ActionAnalyzer()

    action_analyzer.initialize(config)

    # 视频文件路径
    video_paths = get_video_path('test_video_zg')

    output_file_path = 'results5'

    for input_file in video_paths:
        assert os.path.isfile(input_file), f"{input_file} not exists"
        print(input_file)
        # json_file = input_file[:-4] + '_moc.json'
        # ref_info ,ball_trace = get_ref_info(json_file)  # 获取参考信息(即篮球位置信息)
        ref_info = {'shot_index': None, 'ball_bbox': None}

        cap = cv2.VideoCapture(input_file)

        while cap.isOpened():
            ret, frame_rgb = cap.read()
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 如果没有读取到帧，则退出循环
            if not ret:
                # 视频分析完成，获取结果并reset模型参数
                result = action_analyzer.get_results(ref_info)
                action_analyzer.reset()

                # 可视化
                vis_2(input_file, output_file_path, result, config, ref_info , ball_trace=None)
                break

            # if frame_index >= frame_range[0] and frame_index <= frame_range[1] and frame_index % config['step'] == 0:
            # print(frame_index)
            if frame_index % config['step'] == 0:
                # 处理帧
                action_analyzer.process(frame_rgb, frame_index)
                print(frame_index)
