import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("/home/zhangwenjing/图片/show.png")
plt.imshow(img)

plt.show()