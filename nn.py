from engine import Atom
import random


def mse_loss(targets: list[Atom], predictions: list[Atom]):
	loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(targets, predictions)])
	return loss


class Mod:
	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0

	def parameters(self):
		return []


class Neuron(Mod):
	'''
	A neuron is a function that takes in a list of atoms and returns a single atom.

	nin (int): number of inputs
	'''
	def __init__(self, nin: int):
		self.weights = [Atom(random.uniform(-1, 1)) for _ in range(nin)]
		self.bias = Atom(random.uniform(-1, 1))

	def __call__(self, x: list[Atom]):
		# weights * x + bias
		act = sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
		return act.relu()

	def parameters(self):
		return self.weights + [self.bias]


class Layer(Mod):
	'''
	A layer is a list of neurons.

	nin (int): number of inputs
	n_out (int): number of outputs
	'''
	def __init__(self, nin: int, n_out: int):
		self.neurons = [Neuron(nin) for _ in range(n_out)]

	def __call__(self, x: list[Atom]):
		results = [n(x) for n in self.neurons]
		return results[0] if len(results) == 1 else results

	def __repr__(self):
		return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

	def parameters(self):
		params = []
		for neuron in self.neurons:
			ps = neuron.parameters()
			params.extend(ps)
		return params


class MLP(Mod):
	'''
	A multilayer perceptron is a list of layers.

	nin (int): number of inputs
	nouts (list[int]): number of outputs in each layer
	'''
	def __init__(self, nin: int, nouts: list[int]):
		sizes = [nin] + nouts
		self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

	def __call__(self, x: list[Atom]):
		for layer in self.layers:
			x = layer(x)
		return x

	def __repr__(self):
		return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

	def parameters(self):
		params = []
		for layer in self.layers:
			ps = layer.parameters()
			params.extend(ps)
		return params
