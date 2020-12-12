import cv2
import numpy as np

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)

def nothing(*arg):
    pass

cv2.namedWindow("hik_vision")  # создаем главное окно
cv2.namedWindow("result")  # создаем главное окно

cv2.namedWindow("settings")  # создаем окно настроек

cv2.createTrackbar('Gauss', 'settings', 2, 4, nothing)
cv2.createTrackbar('Canny', 'settings', 1, 10, nothing)

# def scaleImage(image, scale_percent):
#    width = int(image.shape[1] * scale_percent / 100)
#    height = int(image.shape[0] * scale_percent / 100)
#    dim = (width, height)
#    image = cv2.resize(image, dim, interpolatiqon = cv2.INTER_AREA)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 24) # Частота кадров
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Ширина кадров в видеопотоке.
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Высота кадров в видеопотоке.

#url = 'rtsp://192.168.1.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4'
#url = 'rtsp://192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264'
url = 'rtsp://admin:pP@697469@192.168.1.102:554/Streaming/Channels/101'
cap = cv2.VideoCapture(url)
#filename = 'cap2.mp4'
#filename = 'cap.avi'
#cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#codec = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('captured.avi',codec, 25.0, (width,height))
#codec = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('captured.mp4',codec, 25.0, (width,height))

kernel_size: int = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
#gauss = cv2.getGaussianKernel(kernel_size, 0)
#gauss = gauss * gauss.transpose(1, 0)

while (cap.isOpened()):

    ret, frame = cap.read()

    gs = cv2.getTrackbarPos('Gauss', 'settings')
    gs = gs * 2 + 1
    cn = cv2.getTrackbarPos('Canny', 'settings')
    cn = cn * 10

    img = cv2.GaussianBlur(frame, (gs, gs), 1.5)
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #img = cv2.erode(frame, kernel, iterations=1)
    #img = cv2.dilate(frame, kernel, iterations=1)

    edge = cv2.Canny(img, cn, cn)
    #img = cv2.dilate(img, kernel)
    #edges = contrast_stretch(edges)


    #Вывод изображений
    #cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    #cv2.rectangle(img, (50, 50), (511, 511), (0, 0, 255), 5)
    #img = cv2.add(img, edge)
    #img = cv2.bitwise_xor(img, edge, mask=None) # false
    cv2.imshow('hik_vision',img)
    #viewImage(img, 'hik_vision')
    #out.write(img)
    cv2.imshow('result',edge)
    #viewImage(edge, 'result')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#out.release()
cap.release()
cv2.destroyAllWindows()

#
#
#import numpy as np, cv2
#
#img1 = cv2.imread(fn1, 0)
#img2 = cv2.imread(fn2, 0)
#h1, w1 = img1.shape[:2]
#h2, w2 = img2.shape[:2]
#vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
#vis[:h1, :w1] = img1
#vis[:h2, w1:w1+w2] = img2
#vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

#cv2.imshow("test", vis)
#cv2.waitKey()
