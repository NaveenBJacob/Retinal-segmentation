from PIL import Image
ImagePath="./IDRiD_01.jpg"
ImageFile= Image.open(ImagePath)
size = 128,128
ImageFile.thumbnail(size)
ImageFile.show()
