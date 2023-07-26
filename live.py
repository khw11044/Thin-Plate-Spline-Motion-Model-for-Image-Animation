import cv2
from argparse import ArgumentParser
import numpy as np
import imageio
import torch
from skimage.transform import resize
import yaml
from tqdm import tqdm
from scipy.spatial import ConvexHull
from skimage import img_as_ubyte
# import copy

from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork


def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network

# def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative'):
#     assert mode in ['standard', 'relative', 'avd']
#     with torch.no_grad():
#         predictions = []
#         source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
#         source = source.to(device)
#         driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
#         kp_source = kp_detector(source)
#         kp_driving_initial = kp_detector(driving[:, :, 0])

#         for frame_idx in tqdm(range(driving.shape[2])):
#             driving_frame = driving[:, :, frame_idx]
#             driving_frame = driving_frame.to(device)
#             kp_driving = kp_detector(driving_frame)
#             if mode == 'standard':
#                 kp_norm = kp_driving
#             elif mode=='relative':
#                 kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
#                                     kp_driving_initial=kp_driving_initial)
#             elif mode == 'avd':
#                 kp_norm = avd_network(kp_source, kp_driving)
#             dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
#                                                     kp_source=kp_source, bg_param = None, 
#                                                     dropout_flag = False)
#             out = inpainting_network(source, dense_motion)

#             predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
#     return predictions

def make_animation(source_image, driving_frame, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        # source = source.to(device)
        driving = torch.tensor(driving_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        
        # driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving)

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
                
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='./config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='./checkpoints/vox.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='./my_sources/woman2.jpg', help="path to source image")
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'], help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")
    parser.add_argument("--recode", default=True, help="recode video")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    opt = parser.parse_args()
    mode = opt.mode
    
    
    source_image = imageio.imread(opt.source_image)
    
    if opt.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    source_image = resize(source_image, opt.img_shape)[..., :3] # (256, 256, 3)
    re_source_image = source_image.copy().astype(np.float32)
    re_source_image = cv2.cvtColor(re_source_image, cv2.COLOR_BGR2RGB)
    inpainting_network, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = opt.config, checkpoint_path = opt.checkpoint, device = device)
    
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        kp_source = kp_detector(source)
        
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       # 또는 cap.get(3)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)     # 또는 cap.get(4)
    print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
    print('resize width, height', opt.img_shape)
    if opt.recode:
        file_path = './' + opt.source_image.split('/')[-1].split('.')[0] + '.avi'
        FPS = 20
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')   
        # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        # size = (int(width)*3, int(height))                   # 프레임 크기
        # out_video_recoder = cv2.VideoWriter(file_path, fourcc, fps, size)
        frames_for_video=[]
    
    current_frame=0
    while(True):
        ret, frame = cap.read()    # Read 결과와 frame

        if(ret) :
            frame = resize(frame, opt.img_shape)[..., :3]
            driving = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
            kp_driving = kp_detector(driving)
            if current_frame==0:
                kp_driving_initial = kp_driving.copy()
                kp_norm = kp_driving
            else:
                if mode == 'standard':
                    kp_norm = kp_driving
                elif mode=='relative':
                    kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                        kp_driving_initial=kp_driving_initial)
                elif mode == 'avd':
                    kp_norm = avd_network(kp_source, kp_driving)

            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param = None, 
                                                dropout_flag = False)
            out = inpainting_network(source, dense_motion)
            prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            # prediction = prediction.astype(np.float64)
            prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
            # output = img_as_ubyte(prediction) 
            attached_images = np.hstack((frame,prediction))
            attached_images = np.hstack((re_source_image, attached_images))
            # out_video_recoder.write((attached_images).astype(np.uint8))
            current_frame+=1 
            if opt.recode:
                frames_for_video.append(attached_images[...,::-1])
            cv2.imshow('original_frame', attached_images)
            # cv2.imshow('original_frame', frame)         # 컬러 화면 출력
            # cv2.imshow('deepfake_frame', prediction)    # 컬러 화면 출력  # 따로 저장하려면 frame이랑 prediction 각각 저장 

            if cv2.waitKey(1) == ord('q'):
                break
    if opt.recode:
        imageio.mimsave(file_path, [img_as_ubyte(frame) for frame in frames_for_video], fps=FPS)
        # for i in range(len(frames_for_video)):
        #     out_video_recoder.write((frames_for_video[i]*255).astype(np.uint8))
            # out_video_recoder.write(frames_for_video[i])
        # out_video_recoder.release()
    cap.release()
    cv2.destroyAllWindows()