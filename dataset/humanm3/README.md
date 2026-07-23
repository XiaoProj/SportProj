[Human-M3 dataset](https://arxiv.org/abs/2308.00628.pdf)

Folder structure:
test (train)
    - basketball1
    - basketball2
        - camera_calibration # Camera calibration matrix
        - images # Named as sampled timestamp, sorted by sequence
        - pointcloud
        - pose_calib # pose annotation with 15 joints. Named as:
             ( 'pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'neck', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
        - smpl_estimated 
            Estimated pesudo SMPL labels by [Smplify-cloud](https://arxiv.org/pdf/2311.11971.pdf).
    - intersection
    - plaza