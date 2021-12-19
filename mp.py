from multiprocessing import Pool
import os
import argparse
class Engine(object):
	def __init__(self, args):
		print('init engine')
		self.args = args

	def __call__(self, OCEAN_TASK_ID):
		run(self.args, OCEAN_TASK_ID)

def run(args, OCEAN_TASK_ID):
	the_command = 'python train_modality.py' \
	              + ' --OCEAN_id=' + str(OCEAN_TASK_ID) \
	              + ' --setting=' + str(args.setting)
	print(the_command)
	os.system(the_command)

def main(args):

	OCEAN_TASK_IDs = [i for i in range(5)]

	try:
		pool = Pool(5)
		engine = Engine(args)
		pool.map(engine, OCEAN_TASK_IDs)

	finally:
		pool.close()
		pool.join()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--setting", type=str)
	args = parser.parse_args()
	print(args)
	main(args)