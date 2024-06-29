import math
import random


class Atom:
	def __init__(self, data, _children=(), _op='', label=''):
		self.data = data
		self._prev = set(_children)
		self._op = _op
		self.label = label
		self.grad = 0.0
		self._backward = lambda: None  # Default is a function that does nothing

	def __repr__(self):
		return f"Atom(data={self.data})"

	def __add__(self, other):
		other = other if isinstance(other, Atom) else Atom(other)
		result = Atom(self.data + other.data, (self, other), '+')

		def _backward():
			self.grad += 1.0 * result.grad
			other.grad += 1.0 * result.grad

		result._backward = _backward
		return result

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		return self + (-other)

	def __neg__(self):
		return self * -1

	def __mul__(self, other):
		other = other if isinstance(other, Atom) else Atom(other)
		result = Atom(self.data * other.data, (self, other), "*")

		def _backward():
			self.grad += other.data * result.grad
			other.grad += self.data * result.grad

		result._backward = _backward
		return result

	def __rmul__(self, other):
		return self * other

	def __pow__(self, other):
		assert isinstance(other, (int, float))
		result = Atom(self.data ** other, (self,), f'**{other}')

		def _backward():
			self.grad += other * (self.data ** (other - 1)) * result.grad

		result._backward = _backward
		return result

	def __truediv__(self, other):
		return self * other ** -1

	def tanh(self):
		x = self.data
		t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
		result = Atom(t, (self,), 'tanh')

		def _backward():
			self.grad += (1 - t ** 2) * result.grad

		result._backward = _backward
		return result

	def exp(self):
		x = self.data
		result = Atom(math.exp(x), (self,), 'exp')

		def _backward():
			self.grad += result.data * result.grad

		result._backward = _backward
		return result

	def backward(self):
		topo = []
		visited = set()

		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)

		build_topo(self)
		self.grad = 1.0
		for n in reversed(topo):
			n._backward()
