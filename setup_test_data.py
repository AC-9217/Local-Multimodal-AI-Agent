# 生产测试样例
import fitz
from PIL import Image
import os


def create_image_red():
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save("test_image_1.jpg")
    print("Created test_image_1.jpg")
def create_image_blue():
    img = Image.new('RGB', (100, 100), color = 'blue')
    img.save("test_image_2.jpg")
    print("Created test_image_2.jpg")
def create_image_green():
    img = Image.new('RGB', (100, 100), color = 'green')
    img.save("test_image_3.jpg")
    print("Created test_image_3.jpg")
def create_image_yellow():
    img = Image.new('RGB', (100, 100), color = 'yellow')
    img.save("test_image_4.jpg")
    print("Created test_image_4.jpg")

if __name__ == "__main__":
    create_image_red()
    create_image_blue()
    create_image_green()
    create_image_yellow()   
