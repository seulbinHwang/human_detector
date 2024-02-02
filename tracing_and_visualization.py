import cv2
import numpy as np


# 클릭 이벤트가 발생할 때 호출되는 콜백 함수
def get_color_by_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_color = image[y, x]
        hsv_pixel_color = \
        cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # HSV 색상 범위 설정
        offset = 10
        lower_bound = np.array([max(hsv_pixel_color[0] - offset, 0), 100, 100])
        upper_bound = np.array(
            [min(hsv_pixel_color[0] + offset, 179), 255, 255])

        # 마스크 생성 및 조끼 픽셀 추출
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        vest_pixels = cv2.bitwise_and(image, image, mask=mask)

        # 추출된 조끼 픽셀을 화면에 표시
        cv2.imshow('Vest Pixels', vest_pixels)


# 이미지 불러오기
image_path = 'your_image_path.jpg'
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 'image'라는 이름의 윈도우 생성 및 콜백 함수 설정
cv2.namedWindow('image')
cv2.setMouseCallback('image', get_color_by_click)

# 원본 이미지 표시
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()