import pytest
import torch

from engine import Atom, Matrix

def approx_equal(val1, val2, rel_tol=1e-6):
    return abs(val1 - val2) <= rel_tol * max(abs(val1), abs(val2))

def test_atom_add():
    a = Atom(2.0)
    b = Atom(3.0)
    c = a + b

    # PyTorch verification
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch + b_torch

    assert approx_equal(c.data, c_torch.item())

    c.grad = 1.0
    c.backward()
    c_torch.backward()

    assert approx_equal(a.grad, a_torch.grad.item())
    assert approx_equal(b.grad, b_torch.grad.item())

def test_atom_mul():
    a = Atom(2.0)
    b = Atom(3.0)
    c = a * b

    # PyTorch verification
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch * b_torch

    assert approx_equal(c.data, c_torch.item())

    c.grad = 1.0
    c.backward()
    c_torch.backward()

    assert approx_equal(a.grad, a_torch.grad.item())
    assert approx_equal(b.grad, b_torch.grad.item())

def test_atom_neg():
    a = Atom(2.0)
    b = -a

    # PyTorch verification
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = -a_torch

    assert approx_equal(b.data, b_torch.item())

    b.grad = 1.0
    b.backward()
    b_torch.backward()

    assert approx_equal(a.grad, a_torch.grad.item())

def test_atom_tanh():
    a = Atom(2.0)
    b = a.tanh()

    # PyTorch verification
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = a_torch.tanh()

    assert approx_equal(b.data, b_torch.item())

    b.grad = 1.0
    b.backward()
    b_torch.backward()

    assert approx_equal(a.grad, a_torch.grad.item())

def test_atom_exp():
    a = Atom(2.0)
    b = a.exp()

    # PyTorch verification
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = a_torch.exp()

    assert approx_equal(b.data, b_torch.item())

    b.grad = 1.0
    b.backward()
    b_torch.backward()

    assert approx_equal(a.grad, a_torch.grad.item())

def test_matrix_add():
    A_data = [[1.0, 2.0], [3.0, 4.0]]
    B_data = [[5.0, 6.0], [7.0, 8.0]]
    A = Matrix(A_data)
    B = Matrix(B_data)
    C = A + B

    # PyTorch verification
    A_torch = torch.tensor(A_data, requires_grad=True)
    B_torch = torch.tensor(B_data, requires_grad=True)
    C_torch = A_torch + B_torch

    for i in range(2):
        for j in range(2):
            assert approx_equal(C.data[i][j].data, C_torch[i, j].item())

    C.backward()
    C_torch.backward(torch.ones_like(C_torch))

    for i in range(2):
        for j in range(2):
            assert approx_equal(A.data[i][j].grad, A_torch.grad[i, j].item())
            assert approx_equal(B.data[i][j].grad, B_torch.grad[i, j].item())

def test_matrix_mul():
    A_data = [[1.0, 2.0], [3.0, 4.0]]
    B_data = [[5.0, 6.0], [7.0, 8.0]]
    A = Matrix(A_data)
    B = Matrix(B_data)
    C = A * B

    # PyTorch verification
    A_torch = torch.tensor(A_data, requires_grad=True)
    B_torch = torch.tensor(B_data, requires_grad=True)
    C_torch = A_torch * B_torch

    for i in range(2):
        for j in range(2):
            assert approx_equal(C.data[i][j].data, C_torch[i, j].item())

    C.backward()
    C_torch.backward(torch.ones_like(C_torch))

    for i in range(2):
        for j in range(2):
            assert approx_equal(A.data[i][j].grad, A_torch.grad[i, j].item())
            assert approx_equal(B.data[i][j].grad, B_torch.grad[i, j].item())

def test_matrix_matmul():
    A_data = [[1.0, 2.0], [3.0, 4.0]]
    B_data = [[5.0, 6.0], [7.0, 8.0]]
    A = Matrix(A_data)
    B = Matrix(B_data)
    C = A.matmul(B)

    # PyTorch verification
    A_torch = torch.tensor(A_data, requires_grad=True)
    B_torch = torch.tensor(B_data, requires_grad=True)
    C_torch = A_torch @ B_torch

    for i in range(2):
        for j in range(2):
            assert approx_equal(C.data[i][j].data, C_torch[i, j].item())

    C.backward()
    C_torch.backward(torch.ones_like(C_torch))

    for i in range(2):
        for j in range(2):
            assert approx_equal(A.data[i][j].grad, A_torch.grad[i, j].item())
            assert approx_equal(B.data[i][j].grad, B_torch.grad[i, j].item())

if __name__ == "__main__":
    pytest.main()
