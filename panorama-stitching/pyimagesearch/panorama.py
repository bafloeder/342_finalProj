# import the necessary packages
import numpy as np
import imutils
import cv2
import multi_band_blending
import numpy as np


def warpTwoImages(img1, img2, H):
	'''warp img2 to img1 with homograph H'''
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
	pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin, -ymin]
	Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
	result = cv2.warpPerspective(img1, Ht.dot(H), (xmax - xmin, ymax - ymin))

	PercentWidth = 0.95
	for i in range(img2.shape[1]):
		for j in range(img2.shape[0]):
			if i >= PercentWidth * img2.shape[1]:
				if result[j][i][0] > 0 and result[j][i][1] > 0 and result[j][i][2] > 0:
					result[j][i] = (0.5 * result[j][i] + 0.5 * img2[j][i])

			else:
				result[j][i] = img2[j][i]




	# result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img2
	return result

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageLeft, imageMid, imageRight) = images
		(kpsM, featuresMid) = self.detectAndDescribe(imageMid)
		(kpsL, featuresLeft) = self.detectAndDescribe(imageLeft)
		(kpsR, featuresRight) = self.detectAndDescribe(imageRight)


		# match features between the two images
		M = self.matchKeypoints(kpsL, kpsM,
			featuresLeft, featuresMid, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		# warpedImageLeft = cv2.warpPerspective(imageLeft, H,
		# 	(imageLeft.shape[1] + imageMid.shape[1]+2000, imageLeft.shape[0]), flags=cv2.WARP_INVERSE_MAP)
		warpedImageLeft = warpTwoImages(imageLeft, imageMid, H)
		# result = multi_band_blending.multi_band_blending(imageB, warpedImageA, 30)

		(kpsM, featuresMid) = self.detectAndDescribe(warpedImageLeft)

		M = self.matchKeypoints(kpsR, kpsM,
			featuresRight, featuresMid, ratio, reprojThresh)

		(matches, H, status) = M

		warpedImageRight = cv2.warpPerspective(imageRight, H, (imageRight.shape[1] + warpedImageLeft.shape[1], imageRight.shape[0]+750))

		result = warpedImageRight

		PercentWidth = 0.95
		for i in range(warpedImageLeft.shape[1]):
			for j in range(warpedImageLeft.shape[0]):
				if i >= PercentWidth*warpedImageLeft.shape[1]:
					if result[j][i][0] > 0 and result[j][i][1] > 0 and result[j][i][2] > 0:
						result[j][i] = (0.5*result[j][i] + 0.5*warpedImageLeft[j][i])

				else:
					result[j][i] = warpedImageLeft[j][i]



		# result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB







		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageLeft, imageMid, kpsL, kpsM, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis

