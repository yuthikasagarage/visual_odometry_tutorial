import time
import numpy as np 
import cv2
from demo_superpoint import SuperPointNet
import torch
from raftoptical import *
                
                
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))



feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )



color = np.random.randint(0, 255, (100, 3))

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.trajectories = []
        self.frame_idx = 0
        self.cur_R = None
        self.orb = orb = cv2.SIFT_create(nfeatures=6000)
        self.cur_t = None
        self.mask = None
        self.px_ref = None
        self.px_cur = None
        self.mask =[]
        self.depth_map = []
        self.focal = cam.fx
        self.fps = 0
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        # Move model to GPU if available    
        self.device = torch.device("cuda")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.parser = ArgumentParser()
        self.parser.add_argument("--model", help="restore checkpoint", default="./raft-small.pth")
        self.parser.add_argument("--small", action="store_true", help="use small model", default='small')
        self.parser.add_argument("--iters", type=int, default=12)
        self.parser.add_argument(
            "--mixed_precision", action="store_true", help="use mixed precision"
        )
        self.args = self.parser.parse_args()
        # Load transforms to resize and normalize the image
        self.model = RAFT(self.args)
        self.pretrained_weights = torch.load(self.args.model)
        self.nnModel = torch.nn.DataParallel(self.model)
        self.nnModel.load_state_dict(self.pretrained_weights)
        

    def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
    
    def featureTracking(self, image_ref, image_cur, px_ref):
        # trajectory_len = 40
        # detect_interval = 5
   
        # start = time.time()
        # img = image_cur.copy()
        # kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
        # st = st.reshape(st.shape[0])
        # kp1 = px_ref[st == 1]
        # kp2 = kp2[st == 1]
        # def featureTracking(self, prev_img, curr_img):


        # find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(image_ref, None)
        kp2, des2 = self.orb.detectAndCompute(image_cur, None)

        # use brute-force matcher
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        # Match ORB descriptors
        matches = bf.match(des1, des2)

        # Sort the matched keypoints in the order of matching distance
        # so the best matches came to the front
        matches = sorted(matches, key=lambda x: x.distance)

        img_matching = cv2.drawMatches(image_ref, kp1, image_cur, kp2, matches[0:100], None)
        cv2.imshow('feature matching', img_matching)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]) 
        return pts1, pts2
    
    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME 
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if(absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
            self.cur_R = R.dot(self.cur_R)
        if(self.px_ref.shape[0] < kMinNumFeature):
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur
    
    def inference(self,img, img2):
         # load pretrained weight
        self.nnModel.to(self.device)
        # change model's mode to evaluation
        self.nnModel.eval()
        # frame preprocessing
        frame_1 = frame_preprocess(img, self.device)
        frame_2 = img2
        
        with torch.no_grad():
                # read the next frame
            frame_2 = frame_preprocess(frame_2, self.device)
                # predict the flow
            flow_low, flow_up = self.nnModel(frame_1, frame_2, iters=12, test_mode=True)
                # transpose the flow output and convert it into numpy array
            vizualize_flow(frame_1, flow_up)

    
    def update(self, img, frame_id, img2):
  
        # assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        # self.new_frame = img
        # if(self.frame_stage == STAGE_DEFAULT_FRAME):
        #     self.processFrame(frame_id)
        #     self.mask = np.zeros_like(self.new_frame)
        # elif(self.frame_stage == STAGE_SECOND_FRAME):
        #     self.processSecondFrame()
        #     self.mask = np.zeros_like(self.new_frame)
        # elif(self.frame_stage == STAGE_FIRST_FRAME):
        #     self.processFirstFrame()
        # self.last_frame = self.new_frame   
        # self.mask = np.zeros_like(self.new_frame)
        
        imgMidas = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        start = time.time()
        self.midas.to(self.device)
        self.midas.eval()
        transform = self.midas_transforms.small_transform
        input_batch = transform(imgMidas).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=imgMidas.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        self.depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


        end = time.time()
        totalTime = end - start

        self.fps = 1 / totalTime

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
        
        
        # optical flow 
        
        imgOp = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img2Op = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        
        y=0
        x=351
        h=360
        w=640
        imgOp = imgOp[y:y+h, x:x+w]
        img2Op = img2Op[y:y+h, x:x+w]
        
        self.inference(imgOp, img2Op)
