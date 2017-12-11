import os
import struct
import time
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import zmq
import pygame
import skimage.filters.rank as sfr
from pyg_viewer import *
from skimage.morphology import disk
from collections import deque
from hand_segment import *
ref_point = []
img_refresh = None

img_pp = None
img_dist = None

READ_MODE = True  # False
Q = deque(maxlen=20)

# def on_click(event, x, y, flags, param):
#     global ref_point, img_refresh
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         ref_point = [(x, y)]
#         x = x >> 1
#         y = y >> 1
#         if img_dist is not None and x < img_dist.shape[1] and y < img_dist.shape[0]:
#             print("(%03d,%03d)=%.2f mm" % (x, y, img_dist[y, x] / 10))
#
#     pass

# 预处理
def preprocess_py_v2(img):

    # pre filter the image
    img_src_f, max_contour = pre_filter(img)
    # cv2.drawContours(img_src_f, [max_contour], 0, (255, 128, 0), 2, 4)
    # cv2.imshow("contour",img_src_f)
    # print("********************************")
    # print(max_contour)
    # print("********************************")

    img_pp = cv2.cvtColor(img_src_f, cv2.COLOR_GRAY2BGR)
    # img_logo = cv2.imread("sjtu.jpg")
    # cv2.drawContours(img_pp, [max_contour], 0, (0,0,255), 2)
    # cv2.imshow("contour", img_pp)
    img_bound_wh = (img_src_f.shape[1], img_src_f.shape[0])
    # if max_contour is None:
    #     return None
    # --- calc palm center
    init_point, init_r, bstatus = find_init_search_point(max_contour, img_bound_wh)

    if init_point is None:
        print(" invalid pos!")
        return None
    # init_point = centroid
    # init_r = 2.0
    # print(init_point, init_r)
    c, r = circle_area_mean_shift(max_contour, init_point, init_r=init_r,
                                  min_converge=1.0, area_ratio_threshold=0.6, radius_step=2.0, img_dump=img_pp)
    # print(c, r)

    # calc updated contour
    new_contour = update_contour(max_contour, c, r, img_bound_wh, init_point)

    if new_contour is None:
        return None

    hull_points = cv2.convexHull(new_contour)

    if hull_points is None:
        return None

    # ----- visualization to img_dump
    # draw contour/ palm circle and centroid
    contour_moment = cv2.moments(max_contour)
    centroid = (int(contour_moment["m10"] / contour_moment["m00"]),
                int(contour_moment["m01"] / contour_moment["m00"]))
    cv2.circle(img_pp, c, int(r), (255, 128, 128), 1)  # parm circle
    cv2.drawContours(img_pp, [max_contour], 0, (255, 128, 0), 2, 4)
    cv2.circle(img_pp, centroid, 1, (128, 128, 0), 1)

    # draw convexHull
    hull = cv2.convexHull(max_contour, False)
    if max_contour is not None and len(max_contour) > 0:
        cv2.drawContours(img_pp, [hull], 0, (128, 255, 0), 2, 4)
    cv2.drawContours(img_pp, [new_contour], 0, (50, 50, 255), 2)
    # # save to file
    # fpath, fname = os.path.split(filename)
    # os.mkdir("%s/preprocess" % fpath) if not os.path.isdir("%s/preprocess" % fpath) else None
    cv2.imshow("pre", img_pp)
    cv2.imwrite("./pre/%s.png"%(datetime.now().strftime("%Y%m%d%H%M%S%f")) , img_pp)

    return np.concatenate((cv2.HuMoments(cv2.moments(new_contour)).flatten(),
                           cv2.HuMoments(cv2.moments(hull_points)).flatten()))


def handle_event(events):
    s_x = 0
    s_y = 0
    for event in events:
        # 选择ROI
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = event.button
            s_x, s_y = event.pos
            print("click")
            # print('起始坐标:(%s,%s)' % (start_x, start_y))
        ## 抬起鼠标
        # el
        #     e_x, e_y = event.pos
            # print('结束坐标:(%s,%s)' % (end_x, end_y))
    return s_x, s_y, True

def main():
    """ main method """
    # os.mkdir("hand-seg")
    global img_refresh, img_pp, img_dist

    frame_cnt = 0
    fps_last_ts = time.time()
    fps_last_fcnt = 0
    fps = 0
    dep_fil = 150
    # # Prepare our context and publisher
    # context = zmq.Context()
    # subscriber = context.socket(zmq.SUB)
    # subscriber.connect("tcp://192.168.7.2:56789")
    # subscriber.setsockopt(zmq.SUBSCRIBE, b"DIST")
    # writer = cv2.VideoWriter("hand_1120.avi", -1, 10.0, (320, 240))
    # cv2.namedWindow("image")
    # # cv2.setMouseCallback("image", on_click)
    if READ_MODE == False:
        file_dep = open("depth1106.bin", 'wb')
    else:
        # file_dep = open("D:\Documents\PyCharm\epc660_data_deamon\epc660_data_deamon/depth1120_far.bin", 'rb')
        # file_dep = open("./new_dataset/wwh-1-1.bin", 'rb')
        file_dep = open("./testdata/testdata/cw-7-0.bin", 'rb')
    viewer = pyg_viewer_c(pan_wid=320, pan_hgt=240, pan=(1, 2), name='手势识别')
    mog = cv2.createBackgroundSubtractorMOG2()
    print("reading....")

    while True:
        # Read envelope with address
        # address, contents = subscriber.recv_multipart()
        sx, sy, state = handle_event(pygame.event.get())
        # update fps
        frame_cnt += 1
        print(frame_cnt)

        cur_ts = time.time()
        if cur_ts > fps_last_ts + 1:
            fps = (frame_cnt - fps_last_fcnt) / (cur_ts - fps_last_ts)
            if cur_ts > fps_last_ts + 3:
                fps_last_ts = cur_ts
                fps_last_fcnt = frame_cnt

        ## 回放模式
        sim_img = np.frombuffer(file_dep.read(2 * 320 * 240), dtype=np.uint16)
        sim_img = sim_img.reshape(240, 320).astype(np.uint16)
        ## 水平翻转
        cv2.flip(sim_img, -1, sim_img)
        cv2.imwrite("save.png", sim_img)
        # sim_img = sfr.minimum(sim_img, disk(1))
        sim_img = cv2.medianBlur(sim_img, 3)
        dep_img = sim_img.copy().astype(np.uint8)


        ## 深度切割阈值确定
        arr = sim_img.flatten()
        arr1 = arr[arr>30]
        # print(len(arr1))
        arr1.sort()
        aver = sum(arr1[:3500]) / 3000
        # print(arr1[:3700])

        if aver < 135:
            print("aver less than 100")
            dep_fil = 135
        else:
            dep_fil = aver
        print("区域的深度平均值=", aver)
        # dep_fil = aver

        sim_img[sim_img > dep_fil] = 0  # 丢弃深度大于115的图像信息
        sim_img = cv2.medianBlur(sim_img, 3)
        # sim_img[sim_img > 115] = 0  # 丢弃深度小于70的图像信息
        # sim_img = sfr.minimum(sim_img, disk(2))

        cal_img = sim_img.copy().astype(np.uint8)
        mask_img = sim_img.copy().astype(np.uint8)

        # cvt_img = cv2.medianBlur(cal_img, 3)
        # cvt_img = cal_img.copy()
        cv2.imshow("sim", cal_img)
        ret, bin_img = cv2.threshold(cal_img, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow("bin", bin_img)
        ## 最大联通区域和空白填充
        i, contour, h = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_contour = None
        for c in contour:
            cur_area = cv2.contourArea(c)
            if max_area < cur_area:
                max_area = cur_area
                max_contour = c
        for c in contour:
            c_min = []
            area = cv2.contourArea(c)
            if area < max_area:#3500:
                c_min.append(c)
                cv2.drawContours(bin_img, c_min, -1, 0, thickness=-1)
        bin_img = cv2.erode(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        bin_img = cv2.dilate(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        cv2.imshow("bin_draw", bin_img)
        print("max contour area is:", max_area)
        if max_area < 1000:
            print("貌似没有人在~")
        else:
            cv2.drawContours(cal_img, max_contour, -1, 255, thickness=2)
        box = cv2.minAreaRect(max_contour)
        cw, ch = box[0]
        width, height = box[1]
        print("cw=%d, ch=%d" % (cw, ch))
        print("width=%d, height=%d"%(width, height))
        # if height > width * 1.8:
        cv2.rectangle(mask_img, (int(cw-width/2-2), int(ch+height*0.1)), (min(320,int(cw+width/2+50)), 240), 0, -1)
        # cv2.imshow("sim", cal_img)
        ## 手部区域提取
        # mask = bin_img.nonzero()
        mask = np.bitwise_and(bin_img>0, bin_img==255)
        mask_img = mask_img * mask
        mask_img = cv2.medianBlur(mask_img, 3)

        cv2.imshow("mask", mask_img)

        cv2.rectangle(bin_img, (int(cw - width / 2 - 2), int(ch + height * 0.1)),
                      (min(320, int(cw + width / 2 + 50)), 240), 0, -1)
        i, clr_contour, h = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        clr_img = cv2.cvtColor(dep_img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(clr_img, clr_contour, -1, (0,0,255), -1)
        cv2.imshow("clr", clr_img)
        cv2.imwrite("./res/%s.png" % (datetime.now().strftime("%Y%m%d%H%M%S%f")), clr_img)
        # img_pp = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)

        #### 金珂毕业论文需要
        # fea = preprocess_py_v2(cal_img)
        # ## 进一步深入精细切割
        # img_bound_wh = (bin_img.shape[1], bin_img.shape[0])
        # init_point, init_r, bstatus = find_init_search_point(max_contour, img_bound_wh)
        # if init_point is None:
        #     print(" invalid pos!")
        # c, r = circle_area_mean_shift(max_contour, init_point, init_r=init_r,
        #                               min_converge=1.0, area_ratio_threshold=0.6, radius_step=2.0, img_dump=bin_img)
        # print(c, r)
        #
        # # calc updated contour
        # new_contour = update_contour(max_contour, c, r, bin_img, init_point)
        # cv2.drawContours(img_pp, [max_contour], 0, (255, 128, 0), thickness=-1)
        # cv2.imshow("refine", img_pp)


        ## 图像裁剪



        ## 轮廓检测
        # ret, bin_img = cv2.threshold(cal_img, 80, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("bin", bin_img)

        ## 录制视频
        # writer.write(sim_img * 200)

        # # 基于单帧的背景扣除
        # if sx>0 and sy>0:
        #     print("点击处图像深度值为：",sim_img[sx, sy])
        # # convert matrix into list
        # arr = cal_img.flatten()
        # arr1 = arr[arr>1]
        # print(len(arr1))
        # arr1.sort()
        # aver = sum(arr1[:4000]) / 2700
        # # print(arr1[:3700])
        # print("区域的深度平均值=", aver)
        #
        # dep_img[dep_img > (aver + 30)] = 0



        # cvt_img = cv2.cvtColor(cal_img, cv2.COLOR_GRAY2RGB)
        # cvt_img = cv2.cvtColor(cvt_img, cv2.COLOR_GRAY2RGB)
        # cvt_img = cvt_img.reshape(240,320,3).astype(np.uint8)

        # n, bins, patches = plt.hist(arr1, bins=20, normed=0, facecolor="green", alpha=0.75)
        # for i in range(500000):


        viewer.update_pan_img_gray(mask_img, pan_id=(0, 0))
        viewer.update_pan_img_gray(dep_img, pan_id=(1, 0))
        # viewer.update_pan_img_rgb(cvt_img, pan_id=(1, 0))
        viewer.update()
        cv2.waitKey(200)

        ## 保存
        # if frame_cnt % 3 == 0:
        #     print("save")
        #     cv2.imwrite("./hand-seg/view/%s.ppm"%(datetime.now().strftime("%Y%m%d%H%M%S%f")),cvt_img)


        # ##  实时采集模式
        # # unpack frame header
        # w, h, d = struct.unpack('=lll', contents[0:12])
        # img = np.frombuffer(file_dep.read(2*320*240), dtype=np.uint16).reshape(240, 320)
        # # 数据保存
        # img_dep = (img /30000 * 255 * 20).astype(np.uint8)
        # img_dep[img_dep > 110] = 0  # 丢弃深度大于110的图像信息
        # img_dep = cv2.medianBlur(img_dep,3)
        #
        # cv2.imwrite("test.png",img_dep)
        # cv2.flip(img_dep, 0, img_dep)
        # cv2.imshow("image", img)
        # cv2.waitKey(30)
        # writer.write(img_dep)
        # 帧率显示
        # img_refresh = (img.astype("float32") / img.max())
        # img_refresh = (img_refresh * 255).astype(np.uint8)
        # img_refresh = cv2.cvtColor(img_refresh, cv2.COLOR_GRAY2BGR)
        # cv2.putText(img_refresh, "FPS: %.1f" % fps, (5, img_refresh.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
        #
        # img_refresh = cv2.resize(img_refresh, None, fx=2, fy=2)
        # cv2.flip(img_refresh, -1, img_refresh)



        user_key = cv2.waitKey(10)

        if user_key == 27:
            break

    # We never get here but clean up anyhow
    # subscriber.close()
    # writer.release()
    file_dep.close()
    # context.term()


if __name__ == "__main__":
    main()
