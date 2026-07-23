"""
Example

M: homography from world to template
Hr: homography from world to image
H: homography from template to image
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import cv2
import json

base = './'
seq = '173833'
idx = 100
src = "seq_{}".format(seq)
folder_images = "images_orig_blurred"
basename = "IMG_{}_frame_{:06d}.JPG".format(seq, idx)

img = imageio.imread(os.path.join(base, src, folder_images, basename))
template = imageio.imread(os.path.join(base, "rectified_template.png"))

poses = json.load(open(os.path.join(base, src, "poses.json")))
Hr = np.array(poses[basename]['Hr'])

M = np.array(json.load(open(os.path.join(base, "homography_rectified_template.json")))['M'])

H = np.dot(Hr, np.linalg.inv(M))

ground_grid = np.array(json.load(open(os.path.join(base, "ground_grid.json"))))

def project_homography(points, H):
    p = np.vstack([points.T, np.ones(len(points))])
    transformed = np.dot(Hr, p)
    return (transformed[:2] / transformed[2]).T

proj_grid = project_homography(ground_grid[:,:2], Hr)

filename_players = os.path.join(base, src, "player_positions.json")
if os.path.isfile(filename_players):
    player_positions = json.load(open(filename_players))
    players = np.array(player_positions[basename])
    
    proj_players = project_homography(players[:,:2], Hr)   
else:
    proj_players = np.empty((0,2))

template_warped = cv2.warpPerspective(template, H, (1920, 1080))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(proj_grid[:,0], proj_grid[:,1], 'r.')
plt.plot(proj_players[:,0], proj_players[:,1], 'c.', markersize=10)
plt.imshow(img)
plt.subplot(1,2,2)
plt.plot(proj_grid[:,0], proj_grid[:,1], 'r.')
plt.plot(proj_players[:,0], proj_players[:,1], 'c.', markersize=10)
plt.imshow(template_warped)
plt.show()