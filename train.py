import caffe
import os


class Training:
	def __init__(self, CNN_NETWORK_PATH, CNN_SOLVER_PATH, USE_GPU=True):
		self.CNN_NETWORK_PATH = CNN_NETWORK_PATH
		self.CNN_SOLVER_PATH = CNN_SOLVER_PATH

		if USE_GPU:
		    caffe.set_device(0)
		    caffe.set_mode_gpu()
		else:
		    caffe.set_mode_cpu()

		self.net = caffe.Net(CNN_NETWORK_PATH, caffe.TRAIN)
		self.solver = caffe.get_solver(CNN_SOLVER_PATH)


	def display_stats(self):
		print("Network layers information:")
		for name, layer in zip(self.net._layer_names, self.net.layers):
		    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
		print("Network blobs information:")
		for name, blob in self.net.blobs.items():
		    print("{:<7}: {}".format(name, blob.data.shape))
		print("self.net.inputs:", self.net.inputs)
		print("self.net.outputs:", self.net.outputs)


	def train(self):
		self.display_stats(self.net)
		self.solver.solve()


def main():
	CNN_NETWORK_PATH = ""
	CNN_SOLVER_PATH = ""
	USE_GPU = True

	training = Training(CNN_NETWORK_PATH, CNN_SOLVER_PATH, USE_GPU=USE_GPU)
	training.train()


if __name__ == "__main__":
	main()