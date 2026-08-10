import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize = False,
    flatten = True
)

img = x_train[0]
label = t_train[0]

print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

image_show(img)