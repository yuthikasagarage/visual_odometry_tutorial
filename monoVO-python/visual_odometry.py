import time
import numpy as np 
import cv2
from demo_superpoint import SuperPointNet
import torch
from raftoptical import *
from run import *
import pandas as pd
                
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
        self.orb = cv2.SIFT_create(nfeatures=6000)
        self.cur_t = None
        self.mask = None
        self.px_ref = None
        self.px_cur = None
        self.mask =[]
        self.depth_map = []
        self.focal = cam.fx
        self.fps = 0
        self.pp = (cam.cx, cam.cy)
        self.calib = pd.read_csv('../dataset-calibration/sequences/00/calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(self.calib.loc['P0:']).reshape((3,4))
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.xfeatures2d.SIFT_create() 
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
        with open(annotations) as f:
            self.annotations = f.readlines()

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
        trajectory_len = 40
        detect_interval = 5
   
        # kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
        # st = st.reshape(st.shape[0])
        # kp1 = px_ref[st == 1]
        # kp2 = kp2[st == 1]
        # pts1= kp1
        # pts2 = kp2

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
        cv2.imshow('sdfsdfsd',self.new_frame)
        cv2.waitKey(3000)
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        print(self.px_ref)
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
    
    def decompose_projection_matrix(self,p):
        '''
        Shortcut to use cv2.decomposeProjectionMatrix(), which only returns k, r, t, and divides
        t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)
        
        Arguments:
        p -- projection matrix to be decomposed
        
        Returns:
        k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector
        
        '''
        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
        t = (t / t[3])[:3]
        
        return k, r, t
    
    def estimate_motion(self, kp1, kp2, k, depth1=None, max_depth=10000):
        """
        Estimate camera motion from a pair of subsequent image frames

        Arguments:
        match -- list of matched features from the pair of images
        kp1 -- list of the keypoints in the first image
        kp2 -- list of the keypoints in the second image
        k -- camera intrinsic calibration matrix 
        
        Optional arguments:
        depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
        max_depth -- Threshold of depth to ignore matched features. 3000 is default

        Returns:
        rmat -- estimated 3x3 rotation matrix
        tvec -- estimated 3x1 translation vector
        image1_points -- matched feature pixel coordinates in the first image. 
                        image1_points[i] = [u, v] -> pixel coordinates of i-th match
        image2_points -- matched feature pixel coordinates in the second image. 
                        image2_points[i] = [u, v] -> pixel coordinates of i-th match
                
        """
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = kp1
        image2_points = kp2

        if depth1 is not None:
            cx = k[0, 2]
            cy = k[1, 2]
            fx = k[0, 0]
            fy = k[1, 1]
            object_points = np.zeros((0, 3))
            delete = []

            # Extract depth information of query image at match points and build 3D positions
            for i, (u, v) in enumerate(image1_points):
              
                z = depth1[int(v), int(u)]*self.getAbsoluteScale(self.frame_idx)
                # If the depth at the position of our matched feature is above 3000, then we
                # ignore this feature because we don't actually know the depth and it will throw
                # our calculations off. We add its index to a list of coordinates to delete from our
                # keypoint lists, and continue the loop. After the loop, we remove these indices
                if z > max_depth:
                    delete.append(i)
                    continue
                    
                # Use arithmetic to extract x and y (faster than using inverse of k)
                x = z*(u-cx)/fx
                y = z*(v-cy)/fy
                object_points = np.vstack([object_points, np.array([x, y, z])])
                # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
                #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])

            image1_points = np.delete(image1_points, delete, 0)
            image2_points = np.delete(image2_points, delete, 0)
            
            # Use PnP algorithm with RANSAC for robustness to outliers
            _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
            #print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))
            
            # Above function returns axis angle rotation representation rvec, use Rodrigues formula
            # to convert this to our desired format of a 3x3 rotation matrix
            rmat = cv2.Rodrigues(rvec)[0]
         
            return rmat, tvec
          
        
        # else:
        #     # With no depth provided, use essential matrix decomposition instead. This is not really
        #     # very useful, since you will get a 3D motion tracking but the scale will be ambiguous
        #     image1_points_hom = np.hstack([image1_points, np.ones(len(image1_points)).reshape(-1,1)])
        #     image2_points_hom = np.hstack([image2_points, np.ones(len(image2_points)).reshape(-1,1)])
        #     E = cv2.findEssentialMat(image1_points, image2_points, k)[0]
        #     _, rmat, tvec, mask = cv2.recoverPose(E, image1_points, image2_points, k)
        
    
    
    
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
            self.vizualize_flow(frame_1, flow_up)

    def vizualize_flow(self, flo):
            # permute the channels and change device is necessary
        # img = img[0].permute(1, 2, 0).cpu().numpy()
        # flo = flo[0].permute(1, 2, 0).cpu().numpy()

            # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
        self.flo = flo
     
        return flo
        
    def update(self, img, frame_id, img2):
  
        # assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
    
        img= cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img2= cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (1024,436), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (1024,436), interpolation=cv2.INTER_LINEAR)
       
        # y=0
        # x=351
        # h=360
        # w=640
        # imgMidas = imgMidas[y:y+h, x:x+w]
        
        # start = time.time()
        # self.midas.to(self.device)
        # self.midas.eval()
        # transform = self.midas_transforms.small_transform
        # input_batch = transform(img).to(self.device)
        
        # with torch.no_grad():
        #     prediction = self.midas(input_batch)

        #     prediction = torch.nn.functional.interpolate(
        #         prediction.unsqueeze(1),
        #         size=img.shape[:2],
        #         mode="bicubic",
        #         align_corners=False,
        #     ).squeeze()

        # depth_map = prediction.cpu().numpy()

        # self.depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        
           

        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # depth_map = (depth_map*255).astype(np.uint8)
        # depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
        
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(img)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(img2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    
        flow = estimate(tenOne, tenTwo)              
        
        flo = self.vizualize_flow(flow.numpy().transpose(1, 2, 0))
        

        cv2.imshow('asa', flo)
        cv2.waitKey(30000)
        # optical flow 
        
        # imgOp = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # img2Op = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        
        # y=0
        # x=351
        # h=360
        # w=640
        # imgOp = imgOp[y:y+h, x:x+w]
        # img2Op = img2Op[y:y+h, x:x+w]
        
        # self.inference(imgOp, img2Op)
     
            #   feature matching
      
        self.new_frame = flo
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
            self.mask = np.zeros_like(self.new_frame)
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
            self.mask = np.zeros_like(self.new_frame)
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame   
        self.mask = np.zeros_like(self.new_frame)
        
        
        
        
        # end = time.time()
        # totalTime = end - start

        # self.fps = 1 / totalTime