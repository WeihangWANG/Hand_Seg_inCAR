## 深度切割阈值确定
        arr = sim_img.flatten()
        arr1 = arr[arr>30]
        # print(len(arr1))
        arr1.sort()
        aver = sum(arr1[:4500]) / 3000
        # print(arr1[:3700])
        print("区域的深度平均值=", aver)
        if aver < 115:
            dep_fil = 115
        else:
            dep_fil = aver


        sim_img[sim_img > dep_fil] = 0  # 丢弃深度大于115的图像信息
        # sim_img[sim_img < 70] = 0  # 丢弃深度小于70的图像信息
        # sim_img = sfr.minimum(sim_img, disk(2))

        cal_img = sim_img.copy().astype(np.uint8)
        mask_img = sim_img.copy().astype(np.uint8)

        cvt_img = cv2.medianBlur(cal_img, 3)
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
            if area < 3000:
                c_min.append(c)
                cv2.drawContours(bin_img, c_min, -1, 0, thickness=-1)
        bin_img = cv2.erode(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        bin_img = cv2.dilate(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        cv2.imshow("bin_draw", bin_img)
        print("max contour area is:", max_area)
        if max_area < 1000:
            print("貌似没有人在~")
        else:
            cv2.drawContours(cal_img, max_contour, -1, 255, thickness=3)
        # cv2.imshow("sim", cal_img)
        ## 手部区域提取
        # mask = bin_img.nonzero()
        mask = np.bitwise_and(bin_img>0, bin_img==255)
        mask_img = mask_img * mask
        mask_img = cv2.medianBlur(mask_img, 3)

        # cv2.imshow("mask", mask_img)