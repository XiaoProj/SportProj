import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import imageio
import cv2
import glob
import json
import pickle

import vis

def json_read(filename):
    with open(filename) as f:    
        data = json.load(f)
    return data

def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

def bbox_from_points(points_2d, pb=0.2, margin=0):
    xmin, ymin = np.nanmin(points_2d, axis=0)
    xmax, ymax = np.nanmax(points_2d, axis=0)
    s = np.mean([xmax-xmin, ymax-ymin])*pb
    xmin = xmin - s - margin
    xmax = xmax + s + margin
    ymin = ymin - s - margin
    ymax = ymax + s + margin
    return xmin, ymin, xmax, ymax

def compute_intersection(bbox, bboxes):
    
    _bboxes = np.array(bboxes)
    
    area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) 
    areas = (_bboxes[:,2] - _bboxes[:,0] + 1) * (_bboxes[:,3] - _bboxes[:,1] + 1) 
    
    xmins = np.maximum(bbox[0], _bboxes[:,0])
    ymins = np.maximum(bbox[1], _bboxes[:,1])
    xmaxs = np.minimum(bbox[2], _bboxes[:,2])
    ymaxs = np.minimum(bbox[3], _bboxes[:,3])

    w = np.maximum(0.0, xmaxs - xmins + 1)
    h = np.maximum(0.0, ymaxs - ymins + 1)
    intersections = w * h

    return (intersections / area).tolist()

def load_poses(base, subjects):
    poses = {}
    for sequence in ['21380_21440_3', '30000_30050_1']:
        for subject in subjects:
            if subject not in poses:
                poses[subject] = {'poses_3d':{'idx_frames':[], 'poses':[]}, 'poses_2d':{}}

            filename = os.path.join(base, "human_poses", sequence, 'pose_subject{}.json'.format(subject))
            if os.path.isfile(filename):
                data = json_read(filename)
                poses[subject]['poses_3d']['idx_frames'] += data['3d']['idx_frame']
                poses[subject]['poses_3d']['poses'] += [np.float32(pose) for pose in data['3d']['pose']]

                for view,x in data['2d'].items():
                    if view not in poses[subject]['poses_2d']:
                        poses[subject]['poses_2d'][view] = {'idx_frames':[], 'poses':[]}
                    poses[subject]['poses_2d'][view]['idx_frames'] += x['idx_frame']
                    poses[subject]['poses_2d'][view]['poses'] += [np.float32(pose) for pose in x['pose']]
    return poses

base = "."
base_images = "multicam"

views = ['ace_{}'.format(i) for i in range(8)]
calibration = json_read(os.path.join(base, "calibration.json"))
trajectories = json_read(os.path.join(base, "trajectories.json"))
subjects = list(trajectories.keys())
poses = load_poses(base, subjects)

idx = 21380
view = 'ace_4'
img = imageio.imread(os.path.join(base_images, "{}/frame_{:06d}.jpg".format(view, idx)))
                     
R = np.array(calibration[view]['R'])
rvec = cv2.Rodrigues(R)[0]
t = np.array(calibration[view]['t'])
K = np.array(calibration[view]['K'])
dist = np.array(calibration[view]['dist'])                    

plt.figure(figsize=(10,10))
plt.title(view)
    
positions = []
for subject,data in trajectories.items():
    if idx in data['indexes']:
        j = data['indexes'].index(idx)
        positions.append(data['positions'][j])

if len(positions):
    proj = cv2.projectPoints(np.float32(positions), rvec, t, K, dist)[0].reshape(-1,2)
    plt.plot(proj[:,0], proj[:,1], 'rx', markersize=6)
    
poses_2d = []
for subject, data in poses.items():
                 
    if idx in data['poses_3d']['idx_frames']:
        j = data['poses_3d']['idx_frames'].index(idx)
        pose_3d = np.array(data['poses_3d']['poses'][j])
        proj = cv2.projectPoints(pose_3d, rvec, t, K, dist)[0].reshape(-1,2)  
                     
        bbox = bbox_from_points(proj, pb=0.0, margin=0)
        
        if compute_intersection(bbox, [(0,0,img.shape[1], img.shape[0])])[0]>0.75:
            poses_2d.append(proj)
                     
img = vis.draw_poses(img, poses_2d, scale=3, rescale=True)     
                     
plt.imshow(img)    
plt.tight_layout()
plt.savefig("example.jpg", dpi=300)