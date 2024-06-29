from engine import Atom
import random


class Mod:
	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0

	def parameters(self):
		return []


class Neuron(Mod):
	def __init__(self, nin):
		self.weights = [Atom(random.uniform(-1, 1)) for _ in range(nin)]
		self.bias = Atom(random.uniform(-1, 1))

	def __call__(self, x):
		# weights * x + bias
		act = sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
		return act.tanh()

	def parameters(self):
		return self.weights + [self.bias]


class Layer(Mod):
	'''
	Just a list of neurons
	'''

	def __init__(self, nin, nout):
		self.neurons = [Neuron(nin) for _ in range(nout)]

	def __call__(self, x):
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
	def __init__(self, nin, nouts):
		sizes = [nin] + nouts
		self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

	def __call__(self, x):
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
