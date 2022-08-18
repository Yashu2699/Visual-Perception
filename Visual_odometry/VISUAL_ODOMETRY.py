import os
impot cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import calculate_error


class Dataset_Handler():

	def __init__(self, sequence, lidar=True, low_memory=True):

		self.lidar = lidar

		self.low_memory = low_memory

		self.seq_dir = '../dataset/sequences/{}/'.format(sequence)
		self.poses_dir = '../dataset/poses/{}.txt'.format(sequence)
		poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)

		self.left_image_files = os.listdir(self.seq_dir + 'image_0')
		self.right_image_files = os.listdir(self.seq_dir + 'image_1')
		self.velodyne_files = os.listdir(self.seq_dir + 'velodyne')
		self.num_frames = len(self.left_image_files)
		self.lidar_path = self.seq_dir + 'velodyne/'

		calib = pd.read_csv(self.seq_dir + 'callib.txt', delimiter=' ', header=None, index_col=0)
		self.P0 = np.array(calib.loc['P0:']).reshape((3,4))
		self.P1 = np.array(calib.loc['P1:']).reshape((3,4))
		self.Tr = np.array(calib.loc['Tr:']).reshape((3,4))


		self.times = np.array([pd.read_csv(self.seq_dir + 'times.txt', delimiter=' ', header=None)])

		# Ground truth poses 
		self.gt = np.zeros((len(poses), 3, 4))

		for i in range(len(poses)):
			self.gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

		if self.low_memory:

			self.reset_frames()

			self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[0], 0)
			self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' + self.right_image_files[0], 0)
			self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1], 0)

			if self.lidar:
				self.first_pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[0], dytpe=np.float32, 
													count=-1).reshape((-1, 4))

			self.imheight = self.first_image_left.shape[0]
			self.imwidth = self.first_image_left.shape[1]


	def reset_frames(self):

		# Resets all generators to the first frames of the sequence
		self.images_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0) for name_left in self.left_image_files)
		self.images_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0) for name_right in self.right_image_files)

		if self.lidar:
			self.pointclouds = (np.fromfile(self.lidar_path + velodyne_file, dtype=np.float32, count=-1).reshape((-1,4)) 
								for velodyne_file in self.velodyne_files)



handler = Dataset_Handler('00')


# Compute left disparity map from a stereo pair of images

def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):

	sad_window = 6
	num_disparities = sad_window * 16
	block_size = 11
	matcher_name = matcher

	if matcher_name == 'bm':
		matcher = cv2.StereoBM_create(num_disparities=num_disparities, block_size=block_size)

	elif matcher_name == 'sgbm':
		matcher = cv2.StereoSGBM_create(num_disparities=num_disparities, minDisparity=0,
										block_size=block_size, P1=8*3*sad_window**2,
										P2=32*3*sad_window**2,
										mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

	start = datetime.datetime.now()

	disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16

	end = datetime.datetime.now()

	if verbose:
		print(f'Time to compute disparity map using Stereo{matcher_name.upper()}:', end-start)

	return disp_left

%matplotlib inline


# using STEREO_BM matcher
disp = compute_left_disparity_map(handler.first_image_left, handler.first_image_right, matcher='bm', verbose=True)
plt.figure(figsize=(11,7))
plt.imshow(disp);


# using STEREO_SGBM matcher
disp = compute_left_disparity_map(handler.first_image_left, handler.first_image_right, matcher='sgbm', verbose=True)
plt.figure(figsize=(11,7))
plt.imshow(disp);


def decompose_projection_matrix(p):

	k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)

	# t is a (4x1 vector) - we make it homogenous
	t = (t / t[3])[:3]

	return k, r, t 

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):

	f = k_left[0][0]

	if rectified:
		b = t_right[0] - t_left[0]

	else:
		b = t_left[0] - t_right[0]



	# set disparity values of -1.0 and 0.0 to a small value(=0.1) so the 
	# resulting depth for these values is large

	disp_left[disp_left == 0.0] = 0.1
	disp_left[disp_left == -1.0] = 0.1

	depth_map = np.ones(disp_left.shape)
	#depth_map = np.ones_like(disp_left)

	depth_map = f * b / disp_left

	return depth_map


k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
k_right, r_right, t_right = decompose_projection_matrix(handler.P1)

depth = calc_depth_map(disp, k_left, t_left, t_right)
plt.figure(figsize=(11,7))
plt.imshow(depth);

plt.hist(depth.flatten())


for i, pixel in enumerate(depth[0]):
	if pixel < depth.max():
		print('First non-max value is at index {}'.format(i))
		break

mask = np.zeros(handler.first_image_left.shape[:2], dtype=np.uint8)
ymax = handler.first_image_left.shape[0]
xmax = handler.first_image_left.shape[1]
cv2.rectangle(mask, (96,0), (xmax, ymax), (255), thickness=-1)
plt.imshow(mask);


#function that computes depth map that uses all other helper functions inside it

def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False, rectified=True):

	disp = compute_left_disparity_map(img_left, img_right, matcher=matcher, rgb=rgb, verbose=verbose)

	k_left, r_left, t_left = decompose_projection_matrix(P0)
	k_right, r_right, t_right = decompose_projection_matrix(P1)

	depth = calc_depth_map(disp, k_left, t_left, t_right)

	return depth

# Feature matching 

def extract_features(image, detector='sift', mask=None):

	if detector == 'sift':
		det = cv2.SIFT_create()

	elif detector == 'orb':
		det = cv2.ORB_create()

	kp, des = det.detectAndCompute(image, mask)

	return kp, des

def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):

	if matching == 'BF':
		if detector == 'sift':
			matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
		elif detector == 'orb':
			matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
		matches = matcher.knnMatch(des1, des2, k=k)

	elif matching == 'FLANN':
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)
		matcher = cv2.FlannBasedMatcher(index_params, search_params)
		matches = matcher.knnMatch(des1, des2, k=k)

	if sort:
		matches = sorted(matches, key = lambda x:x[0].distance)

	return matches 

def filter_matches_distance(matches, dist_threshold):

	filtered_matches = []
	for m, n in matches:
		if m.distance <= dist_threshold * n.distance:
			filtered_matches.append(m)

	return filtered_matches

def visualize_matches(image1, kp1, image2, kp2, match):

	image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
	plt.figure(figsize=(16,6), dpi=100)
	plt.imshow(image_matches)


# Visualizing image matches 
image_left = handler.first_image_left
image_right = handler.first_image_right
image_plus1 = handler.second_image_left

depth = stereo_2_depth(image_left, img_right, handler.P0, handler.P1, matcher='sgbm', verbose=True)
kp0, des0 = extract_features(image_left, 'sift')
kp1, des1 = extract_features(image_plus1, 'sift')

matches = match_features(des0, des1, matching='BF', detector='sift', sort=True)
print('Number of matches before filtering:', len(matches))
matches = filter_matches_distance(matches, 0.3)
print('Number of matches after filtering:', len(matches))
visualize_matches(image_left, kp0, image_plus1, kp1, matches)

# Visualizing image matches 
image_left = handler.first_image_left
image_right = handler.first_image_right
image_plus1 = handler.second_image_left

depth = stereo_2_depth(image_left, img_right, handler.P0, handler.P1, matcher='bm', verbose=True)
kp0, des0 = extract_features(image_left, 'sift')
kp1, des1 = extract_features(image_plus1, 'sift')

matches = match_features(des0, des1, matching='BF', detector='sift', sort=True)
print('Number of matches before filtering:', len(matches))
matches = filter_matches_distance(matches, 0.3)
print('Number of matches after filtering:', len(matches))
visualize_matches(image_left, kp0, image_plus1, kp1, matches)

# Visualizing image matches 
image_left = handler.first_image_left
image_right = handler.first_image_right
image_plus1 = handler.second_image_left

depth = stereo_2_depth(image_left, img_right, handler.P0, handler.P1, matcher='sgbm', verbose=True)
kp0, des0 = extract_features(image_left, 'orb')
kp1, des1 = extract_features(image_plus1, 'orb')

matches = match_features(des0, des1, matching='BF', detector='orb', sort=True)
print('Number of matches before filtering:', len(matches))
matches = filter_matches_distance(matches, 0.3)
print('Number of matches after filtering:', len(matches))
visualize_matches(image_left, kp0, image_plus1, kp1, matches)


# Visualizing lidar pointcloud


%matplotlib notebook

pointcloud  = handler.first_pointcloud
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

xs = pointcloud[:, 0]
ys = pointcloud[:, 1]
zs = pointcloud[:, 2]

ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs))) # set equal scales to all axes
ax.scatter(xs, ys, zs, s=0.01)
ax.grid(False)
ax.axis('off')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(elev=40, azim=180)



# we use the lidar data to compute depths of indices and use these depths
# instead of stereo depths

def pointcloud2image(pointcloud, imheight, imwidth, Tr, P0):

	# Lidar x axis points forward, so we ignore anything with X values less than zero

	pointcloud = pointcloud[pointcloud[:, 0] > 0]

	# Lidar data is (X, Y, X, reflectance). we do not need reflectance value, instead we append 
	# ones along last column for homogeneous calculations

	pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1,1))])

	# Transform pointcloud to camera frame
	cam_xyz = Tr.dot(pointcloud.T)

	# Ignore any point behind camera (already did that but just in case)
	cam_xyz = cam_xyz[:, cam_xyz[2] > 0]

	depth = cam_xyz[2].copy()

	# project the coordinates on a plane at Z=1 by dividing by Z
	cam_xyz /= cam_xyz[2]

	cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])

	# Get pixel coordinates of X, Y, Z points in camera coordinate frame and turn
	# them into integers

	projection = P0.dot(cam_xyz)
	pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')

	# consider only those indices that fit on the image plane

	indices = np.where((pixel_coordinates[:, 0] < imwidth),
						& (pixel_coordinates[:, 0] >= 0),
						& (pixel_coordinates[:, 1] < imheight),
						& (pixel_coordinates[:, 1] >= 0))

	pixel_coordinates = pixel_coordinates[indices]
	depth = depth[indices]

	render = np.zeros((imheight, imwidth))

	for j, (u, v) in enumerate(pixel_coordinates):
		if u >= imwidth or u < 0:
			continue
		if v >= imheight or v < 0:  # already took care of this case in the indices function
			continue

		render[v, u] = depth[j]

	render[render == 0.0] = 3999 # Zero values correspond to the pixel positions where the lidar data is
								 # unavailable

	return render


depth_stereo = stereo_2_depth(image_left, img_right, handler.P0, handler.P1, matcher='sgbm', verbose=True)
depth_lidar = pointcloud2image(handler.first_pointcloud, handler.imheight, handler.imwidth, handler.Tr, handler.P0)
fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(12,8))
ax1.imshow(depth_stereo)
ax2.imshow(depth_lidar)



# Comparison of the stereo depths and lidar depths

idx_height, idx_width = np.where(depth_lidar < 3000)
depth_indx = np.array(list(zip(idx_height, idx_width)))

comparison = np.hstack([depth_stereo[depth_indx[:, 0], depth_indx[:, 1]].reshape(-1,1), 
                        depth_lidar[depth_indx[:, 0], depth_indx[:, 1]].reshape(-1,1)
                       ])
for i, row in enumerate(comparison):
    print('location:', depth_indx[i], 'stereo/lidar depth:', row)
    if i > 100:
        break



def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):

	# Estimate camera motion from a pair of subsequent image frames

	rmat = np.eye(3)
	tvec = np.zeros((3,1))

	image1_points = np.float32([kp1[m.queryIdx].pt for m in match])
	image2_points = np.float32([kp2[m.trainIdx].pt for m in match])


	if depth1 is not None:
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
        object_points = np.zeros((0, 3))
        delete = []


		for i, (u, v) in enumerate(image1_points):
			z= depth1[int(v), int(u)]

			if z > max_depth:
				delete.append(i)
				continue

			x = z * (u - cx) / fx
			y = z * (v - cy) / fy
			object_points = np.vstack([object_points, np.array([x, y, z])])

		image1_points = np.delete(image1_points, delete, 0)
		image2_points = np.delete(image2_points, delete, 0)

		_, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)

		rmat = cv2.Rodrigues(rvec)[0]

	return rmat, tvec, image1_points, image2_points


def visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=None,
					stereo_matcher='bm', mask=None, depth_type='stereo', subset=None, plot=False):
	
	lidar = handler.lidar

	print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector), matching))

    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))

    if lidar:
        print('Improving stereo depth estimation with lidar data')

    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames


   	if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        zs = handler.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')

    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]

    imheight = handler.imheight
    imwidth = handler.imwidth

    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)

    if handler.low_memory:
    	handler.reset_frames()
    	image_plus1 = next(handler.images_left)

    for i in range(num_frames - 1):

    	start = datetime.datetime.now()

    	if handler.low_memory:
    		image_left = image_plus1
    		image_right = next(handler.images_right)
    		image_plus1 = next(handler.images_left)

    	if depth_type == 'stereo':
    		depth = stereo_2_depth(image_left, image_right, P0=handl.P0,
    								P1=handler.P1, matcher=stereo_matcher)
    	else:
    		depth = None


    	# Supercede stereo depth with lidar depth where lidar points are available

    	if lidar:
    		if handler.low_memory:
    			pointcloud = next(handler.pointclouds)

    		else:
    			continue

    		lidar_depth = pointcloud2image(pointcloud, imheight=imheight, imwidth=imwidth,
    										Tr=handler.Tr, P0=handler.P0)

    		indices = np.where(lidar_depth < 3000)

    		depth[indices] = lidar_depth[indices]

    	kp0, des0 = extract_features(image_left, detector, mask)
    	kp1, des1 = extract_features(image_plus1, detector, mask)

    	matches_unfilt = match_features(des0, des1, matching=matching, detector=detector, sort=True)

    	matches = filter_matches_distance(matches_unfilt, filter_match_distance)

    	rmat, tvec, img1_points, img2_points = estimate_motion(matches, kp0, kp1, k_left, depth))

		
		Tmat = np.eye(4)
		Tmat[:3, :3] = rmat
		Tmat[:3, 3] = tvec.T

		T_tot = T_tot.dot(np.linalg.inv(Tmat))

		trajectory[i+1, :, :] = T_tot[:3, :]

		end = datetime.datetime.now()
		print('Time to compute frame {}:'.format(i+1), end-start)
        
        if plot:
            xs = trajectory[:i+2, 0, 3]
            ys = trajectory[:i+2, 1, 3]
            zs = trajectory[:i+2, 2, 3]
            plt.plot(xs, ys, zs, c='chartreuse')
            plt.pause(1e-32)
            
    if plot:        
    plt.close()
        
    return trajectory


handler.lidar = True
start = datetime.datetime.now()
trajectory2 = visual_odometry(handler, 
                              filter_match_distance=0.3, 
                              stereo_matcher='sgbm',
                              mask=mask,
                              subset=None)
end = datetime.datetime.now()
print('Time to perform odometry:', end-start)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory2[:, :, 3][:, 0], 
        trajectory2[:, :, 3][:, 1], 
        trajectory2[:, :, 3][:, 2], label='estimated', color='orange')

ax.plot(handler.gt[:, :, 3][:, 0], 
        handler.gt[:, :, 3][:, 1], 
        handler.gt[:, :, 3][:, 2], label='ground truth')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(elev=-20, azim=270)

calculate_error(handler.gt, trajectory2, 'all')