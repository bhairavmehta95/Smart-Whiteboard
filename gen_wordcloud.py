from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def generate_wordcloud():
	# Read the whole text.
	with open('full_transcript.txt', 'r') as myfile:
		text=myfile.read().replace('\n', '')

	# read the mask / color image

	coloring = imread("3Mcolor.png")

	wc = WordCloud(background_color="white", max_words=2000, mask=coloring,
	               stopwords=STOPWORDS.add("said"))


	# generate word cloud
	wc.generate(text)

	# create coloring from image
	image_colors = ImageColorGenerator(coloring)
	wc.recolor(color_func = image_colors)
	wc.to_file("3MwordcloudColorTDK.png")


if __name__ == '__main__':
	generate_wordcloud()