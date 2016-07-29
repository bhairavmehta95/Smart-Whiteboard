from PIL import Image

im = Image.open('0.jpg')

# will not be neccesary with smaller images
colors = im.getcolors(im.size[0] * im.size[1])
R = 0
G = 0
B = 0
count = len(colors)

for color in colors:
	R += color[1][0]
	G += color[1][1]
	B += color[1][2]

	
print R/count, G/count, B/count
