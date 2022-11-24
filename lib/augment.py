from PIL import Image
from PIL import ImageEnhance

def move(image, offset): 
    return image.offset(offset, 0)

def flip(image):  
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def contrastEnhancement(image, contrast=1.5):  
    enh_con = ImageEnhance.Contrast(image)
    return enh_con.enhance(contrast)

def brightnessEnhancement(image, brightness=1.5): 
    enh_bri = ImageEnhance.Brightness(image)
    return enh_bri.enhance(brightness)

def colorEnhancement(image, color=1.5):  # 颜色增强
    enh_col = ImageEnhance.Color(image)
    return enh_col.enhance(color)
