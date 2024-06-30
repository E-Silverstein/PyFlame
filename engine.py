import math
import random

from numpy import shape


class Atom:
	'''
	An atom is a node in the computation graph.
	It has a data value, a list of children, and an operation.
	'''
	def __init__(self, data: float, _children=(), _op='', label: str=''):
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

	def relu(self):
		result = Atom(max(0, self.data), (self,), 'relu')

		def _backward():
			self.grad += (self.data > 0) * result.grad

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

class Matrix:
    def __init__(self, data):
        self.data = self._setup_data(data)
        self.shape = self._setup_shape(self.data)
        self.grad = [[0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]

    def _setup_data(self, data):
        if isinstance(data[0], Atom):
            return data
        elif isinstance(data[0], float):
            return [Atom(x) for x in data]
        elif isinstance(data[0], list):
            if isinstance(data[0][0], Atom):
                return data
            else:
                return [[Atom(x) for x in row] for row in data]
        else:
            raise ValueError('Invalid data type')
        
    def _setup_shape(self, data):
        if isinstance(data, Atom):
            return (1, 1)
        elif isinstance(data[0], float):
            return (1, len(data))
        else:
            return (len(data), len(data[0]))
        
    def __repr__(self):
        data_str = "[\n"
        for row in self.data:
            data_str += "  " + str(row) + ",\n"
        data_str += "]"
        return f"Matrix(data={data_str}, shape={self.shape})"
    
    def __add__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        assert self.shape == other.shape
        result = Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])])

        def _backward():
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.data[i][j].grad += result.data[i][j].grad
                    other.data[i][j].grad += result.data[i][j].grad

        result._backward = _backward
        return result
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(other)
        assert self.shape == other.shape
        result = Matrix([[self.data[i][j] * other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])])

        def _backward():
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.data[i][j].grad += other.data[i][j].data * result.data[i][j].grad
                    other.data[i][j].grad += self.data[i][j].data * result.data[i][j].grad

        result._backward = _backward
        return result

    def matmul(self, other):
        assert self.shape[1] == other.shape[0]
        result_data = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1])) for j in range(other.shape[1])] for i in range(self.shape[0])]
        result = Matrix(result_data)

        def _backward():
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        self.data[i][k].grad += other.data[k][j].data * result.data[i][j].grad
                        other.data[k][j].grad += self.data[i][k].data * result.data[i][j].grad

        result._backward = _backward
        return result

    def backward(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                self.data[i][j].backward()