from PIL import Image

im = Image.open('fakepath.jpg')

# will not be neccesary with smaller images
colors = im.getcolors(im.size[0] * im.size[1])

print colors[0]