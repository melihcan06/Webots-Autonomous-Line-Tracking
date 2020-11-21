from vehicle import Driver
from controller import Camera, Lidar
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def bas(goruntuler, wait_key=0, baslangic=0, adlar=[]):
    if len(adlar) == 0:
        adlar = [i + 1 for i in range(baslangic, baslangic + len(goruntuler))]
    elif len(goruntuler) > len(adlar):
        fark = len(goruntuler) - len(adlar)
        for i in range(baslangic, baslangic + fark):
            adlar.append(i + 1)
    j = 0
    for i in range(baslangic, baslangic + len(goruntuler)):
        cv2.imshow("grt " + str(adlar[j]), goruntuler[j])
        j += 1

    cv2.waitKey(wait_key)


def hough_transform(img1, rho=1, theta=np.pi / 180, thresh=35, min_theta=0.0, max_theta=np.pi):
    img = img1.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho, theta, thresh, min_theta=min_theta, max_theta=max_theta)

    return lines


def return_road_lines(lines, h, w):
    road_lines = np.zeros((h, w), dtype="uint8")
    if lines is None:
        return road_lines
    count=0
    sum_rho=0
    sum_theta=0
    for idx in range(len(lines)):
        for rho, theta in lines[idx]:
            count+=1
            sum_rho+=rho
            sum_theta+=theta

    mean_rho=sum_rho/count
    mean_theta=sum_theta/count
    a = np.cos(mean_theta)
    b = np.sin(mean_theta)
    x0 = a * mean_rho
    y0 = b * mean_rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(road_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return road_lines

def calc_x_points_hough(w, car_w, reference_height, road_lines):
    x_left = 0
    x_right = w

    right_idx = car_w
    left_idx = car_w
    for j in range(int(w / 2)):
        if x_right == w and road_lines[reference_height][right_idx] == 255:
            x_right = right_idx
        if x_left == 0 and road_lines[reference_height][left_idx] == 255:
            x_left = left_idx
        if x_right != w and x_left != 0:
            break
        right_idx += 1
        left_idx -= 1

    x_mean = int((x_left + x_right) / 2)
    return x_mean, x_right, x_left


def calc_degree(x_mean, car_w, car_h, reference_height):
    # sola doculecekse - li saga dolulecekse + li deger cikmasi icin ayarlandi
    tan = (x_mean - car_w) / (car_h - reference_height)
    degree_line = math.degrees(math.atan(tan))
    # 0 ,180 araligini -0.8 0.8 araligina dagittim
    MAX_DEGREE = 0.8
    degree = MAX_DEGREE * degree_line / 90
    return degree


class preprocess:
    def olcekleme2b(self, girdi, lower_bound=0, upper_bound=255):
        eb = np.max(girdi)
        ek = np.min(girdi)
        ebek = eb - ek
        sinir_fark = upper_bound - lower_bound

        for i in range(girdi.shape[0]):
            for j in range(girdi.shape[1]):
                girdi[i][j] = ((girdi[i][j] - ek) / ebek) * (sinir_fark) + lower_bound

        return girdi

    def olcekleme3b(self, girdi, lower_bound=0, upper_bound=255):
        eb = np.max(girdi)
        ek = np.min(girdi)
        ebek = eb - ek
        sinir_fark = upper_bound - lower_bound

        for i in range(girdi.shape[0]):
            for j in range(girdi.shape[1]):
                girdi[i][j][0] = ((girdi[i][j][0] - ek) / ebek) * (sinir_fark) + lower_bound
                girdi[i][j][1] = ((girdi[i][j][1] - ek) / ebek) * (sinir_fark) + lower_bound
                girdi[i][j][2] = ((girdi[i][j][2] - ek) / ebek) * (sinir_fark) + lower_bound

        return girdi

    # 2 threshold values are used first and their numbers are taken.Used when converting array to img
    def _sinirla(self, a, alt=0, ust=255, veri_tipi="uint8"):
        c = np.clip(a, alt, ust)
        return np.array(c, dtype=veri_tipi)

    # array to img
    def ciktiyi_goruntuye_cevir(self, grt2, olcekleme=True, mutlak_al=False):
        grt = grt2.copy()
        b = grt.shape[0]
        e = grt.shape[1]
        if mutlak_al == True:
            if len(grt.shape) == 3:
                for i in range(b):
                    for j in range(e):
                        grt[i][j][0] = math.fabs(grt[i][j][0])
                        grt[i][j][1] = math.fabs(grt[i][j][1])
                        grt[i][j][2] = math.fabs(grt[i][j][2])
            else:
                for i in range(b):
                    for j in range(e):
                        grt[i][j] = math.fabs(grt[i][j])

        if olcekleme == True:
            if len(grt.shape) == 2:
                grt = self.olcekleme2b(grt)
            else:
                grt = self.olcekleme3b(grt)
        return self._sinirla(grt)


def calc_x_points_unet(reference_height, mask):
    i = reference_height
    x_left = 0
    for j in range(mask.shape[1]):
        if mask[i][j] == 255:
            x_left = j
            break
    x_right = 0
    for j in range(mask.shape[1] - 1, -1, -1):
        if mask[i][j] == 255:
            x_right = j
            break

    x_mean = int((x_left + x_right) / 2)
    return x_mean, x_right, x_left


def color_ranging(prediction, frame_resize):
    hsv = prediction.copy()  # cv2.cvtColor(prediction,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((11, 11))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    road = cv2.bitwise_and(frame_resize[0], frame_resize[0], mask=mask)

    return road, mask


def run_unet(frame):
    frame_resize = cv2.resize(frame, (UNET_SHAPE[0], UNET_SHAPE[1]))
    frame_resize = np.reshape(frame_resize, (1, UNET_SHAPE[0], UNET_SHAPE[1], 3))
    prediction = model.predict(frame_resize)[0]
    prediction = pre.ciktiyi_goruntuye_cevir(prediction)

    return prediction, frame_resize


def mask_bounds(mask,draw=False):
    try:
        y_idxs, x_idxs = np.nonzero(mask)
        max_y = y_idxs[0]
        min_y = y_idxs[0]
        max_x = x_idxs[0]
        min_x = x_idxs[0]
    
        for i in range(len(y_idxs)):
            if y_idxs[i] > max_y:
                max_y = y_idxs[i]
            if y_idxs[i] < min_y:
                min_y = y_idxs[i]
            if x_idxs[i] > max_x:
                max_x = x_idxs[i]
            if x_idxs[i] < min_x:
                min_x = x_idxs[i]
    
        mask2=np.zeros((max_y,max_x),dtype="uint8")
        for i in range(min_y,max_y):
            for j in range(min_x,max_x):
                mask2[i][j]=255
        # w=max_x-min_x
        # h=max_y-min_y
        # return min_y,min_x,h,w
        return min_y, min_x, max_y, max_x, mask2
    except :
        return 0,0,mask.shape[0],mask.shape[1],mask


from tensorflow.keras.models import load_model

UNET_PATH = 'C:/Users/user/Downloads/unet19-20201030T212148Z-001/unet19'
lower_bound = np.array([0, 0, 180])
upper_bound = np.array([50, 40, 255])
# UNET_PATH = 'C:/Users/user/Downloads/unet18-20201030T210359Z-001/unet18'
# lower_bound = np.array([200, 90, 180])
# upper_bound = np.array([255, 180, 255])
model = load_model(UNET_PATH)
UNET_SHAPE = (256, 256, 3)
pre = preprocess()
UNET_ACTIVATE = True

driver = Driver()
timestep = int(driver.getBasicTimeStep())

forward_velocity = 20
brake = 0

camera = driver.getCamera("camera")
Camera.enable(camera, timestep)

# lms291 = driver.getLidar("Sick LMS 291")
# Lidar.enable(lms291, timestep)
# lms291_yatay = Lidar.getHorizontalResolution(lms291)

degree = 0
driver.setSteeringAngle(degree)
counter = 0

while driver.step() != -1:
    driver.setCruisingSpeed(forward_velocity)

    if counter % 30 == 0:
        Camera.getImage(camera)
        Camera.saveImage(camera, "camera.png", 1)
        img1 = cv2.imread("camera.png")

        if UNET_ACTIVATE:#USING FOR ROI
            # w = UNET_SHAPE[1]  # frame.shape[1]
            # h = UNET_SHAPE[0]  # frame.shape[0]
            # car_h, car_w = h, int(h / 2)

            prediction, frame_resize = run_unet(img1)
            road, mask = color_ranging(prediction, frame_resize)

            # reference_height = mask.shape[0] - 50
            # x_mean, x_right, x_left = calc_x_points_unet(reference_height, mask)
            # y_mean = reference_height

            mask = cv2.resize(mask, (img1.shape[1], img1.shape[0]))
            # img=cv2.bitwise_and(img1,img1,mask=mask)
            min_y, min_x, max_y, max_x, mask = mask_bounds(mask,True)
            img = img1[min_y:max_y, min_x:max_x].copy()
            reference_height = int((max_y-min_y)/2)
        else:
            img = img1[300:450, :].copy()
            mask = None
            reference_height = img.shape[0] - 75

        h, w, _ = img.shape
        car_h, car_w = h, int(w / 2)
        
        #reference_height = h-75
        #lines = hough_transform(img)

        lin1 = hough_transform(img, theta=np.pi / 180, min_theta=math.radians(20), max_theta=math.radians(70))#left
        lin2 = hough_transform(img, theta=np.pi / 180, min_theta=math.radians(110), max_theta=math.radians(160))#right
        rl1 = return_road_lines(lin1, h, w)
        rl2 = return_road_lines(lin2, h, w)
        road_lines = cv2.add(rl1, rl2)

        #if lines is not None:

        #road_lines = return_road_lines(lines, h, w)

        x_mean, x_right, x_left = calc_x_points_hough(w, car_w, reference_height, road_lines)

        if x_mean == car_w:  # duz git
            degree = 0
        else:
            degree = calc_degree(x_mean, car_w, car_h, reference_height)

        cv2.circle(img, (car_w, car_h), 20, (0, 255, 255), 2)
        cv2.circle(img, (x_right, reference_height), 20, (0, 0, 255), 2)
        cv2.circle(img, (x_left, reference_height), 20, (0, 0, 255), 2)
        cv2.circle(img, (x_mean, reference_height), 20, (0, 255, 0), 2)
        cv2.line(img, (x_right, reference_height), (x_left, reference_height), (255, 0, 0), 2)
        if mask is None:
            bas([img1, img, road_lines], wait_key=1)
        else:
            bas([img1, img, road_lines, mask], wait_key=1)

        print("degree: ", str(degree), " counter: ", str(counter/30))
    driver.setSteeringAngle(degree)
    counter += 1
