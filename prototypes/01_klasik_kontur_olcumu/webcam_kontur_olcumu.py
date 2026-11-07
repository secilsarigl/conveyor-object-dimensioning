from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to calculate euclidean distance between two points
def euclidean(pt1, pt2):
	return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change if you have multiple cameras

# Reference object dimensions
# Here for reference I have used a 2cm x 2cm square
ref_width_cm = 2

while True:
	# Capture frame-by-frame
	ret, frame = cap.read()
	if not ret:
		break

	# Preprocess frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (9, 9), 0)
	edged = cv2.Canny(blur, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# Find contours
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = [x for x in cnts if cv2.contourArea(x) > 100]

	# Sort contours from left to right as leftmost contour is reference object
	(cnts, _) = contours.sort_contours(cnts)

	# Calculate pixel per cm
	ref_object = cnts[0]
	box = cv2.minAreaRect(ref_object)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	dist_in_pixel = euclidean(tl, tr)
	pixel_per_cm = dist_in_pixel / ref_width_cm

	# Draw contours and dimensions
	for cnt in cnts:
		box = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		(tl, tr, br, bl) = box
		cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)
		wid = euclidean(tl, tr) / pixel_per_cm
		ht = euclidean(tr, br) / pixel_per_cm
		mid_pt_horizontal = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2))
		mid_pt_verticle = (int((tr[0] + br[0]) / 2), int((tr[1] + br[1]) / 2))
		cv2.putText(frame, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
		cv2.putText(frame, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the capture
cap.release()
cv2.destroyAllWindows()
