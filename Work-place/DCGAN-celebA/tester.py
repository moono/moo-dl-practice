import os
import glob
import numpy as np 

datset_dir = '../Data_sets/'
# joined_path = os.path.join(datset_dir, 'img_align_celeba/*.jpg')
# print(joined_path)
# temp = glob.glob( joined_path )
# print(type(temp))
# print(len(temp))

n_data = 7
n_attr = 40
attr_dir = os.path.join(datset_dir, 'celeba_anno')
attr_fn = os.path.join(attr_dir, 'list_attr_celeba_cropped.txt')
# attr_list = np.loadtxt(attr_fn, dtype=int, skiprows=2, usecols=range(1, n_attr + 1))
# print(attr_list)
# print(attr_list.shape)

# with open(attr_fn, 'r') as f:
# 	first_line = f.readline()
# 	print(type(first_line))
# 	print(first_line)
# 	my_data = int(first_line)
# 	print(type(my_data))
# 	print(first_line)
# 	second_line = f.readline()
# 	attr_names = second_line.split()
# 	print(len(attr_names))
# 	print(attr_names)

class AttrParserCelebA(object):
	def __init__(self, attr_fn, convert_to_zeros_and_ones=True):
		'''
		:param attr_fn: attribute text file name
		:param convert_to_zeros_and_ones: converts -1 to 0, 1 to 1
		'''
		# get number of data & atrtibute types
		with open(attr_fn, 'r') as f:
			# parse number of data
			first_line = f.readline()
			self.n_data = int(first_line)

			# parse each attribute names & size
			second_line = f.readline()
			self.attr_names = second_line.split()
			self.n_attr = len(self.attr_names)

		# load whole text file as in np ndarray(2d matrix: n_data x n_attr)
		self.attr_data = np.loadtxt(attr_fn, dtype=int, skiprows=2, usecols=range(1, n_attr + 1))

		# data contains -1: false, 1: true
		# convert that to 0 & 1
		if convert_to_zeros_and_ones:
			self.attr_data = (self.attr_data + 1) // 2


	def get_labels_by_attr_name(self, attr_name):
		# get index of attribute
		try:
			attr_index = self.attr_names.index(attr_name)
		except:
			print('unidentified attribute name...!!!')

		return self.attr_data[:, attr_index]

attr_dir = os.path.join(datset_dir, 'celeba_anno')
attr_fn = os.path.join(attr_dir, 'list_attr_celeba_cropped.txt')
celeba_parser = AttrParserCelebA(attr_fn)
is_male = celeba_parser.get_labels_by_attr_name('Male')
print(is_male.shape)
print(is_male)
