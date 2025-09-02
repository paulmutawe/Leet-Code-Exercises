import numpy as np
import copy
import math
import unittest
from typing import Optional, Tuple

from torch import Value

np.random.seed(42)

class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape
        
    def layer_name(self):
        return self.__class__.__name__
    
    def parameters(self):
        return 0
    
    def forward_pass(self, X, training):
        raise NotImplementedError()
    
    def backward_pass(self, accum_grad):
        raise NotImplementedError()
    
    def output_shape(self):
        raise NotImplementedError
    
class Dense(Layer):
    
    """
    This layer computes:
    
    Y = X @ W + w0
    
    Where:
    
    - X has shape (batch_size, input_dim)
    - W has shape (input_dim, n_units)
    - w0 has shape (1, n_units), broadcast across the batch
    
    """
    
    def __init__(self, n_units: int, input_shape: Optional[Tuple[int, ...]] = None):
        """
        Params
        
        n_units: int 
          Number of neurons / output units in the layer
          
        input_shape: Optional[Tuple[int, ...]] = input_shape
          Expected shape of an input sample 
        
        """
        
        self.layer_input: Optional[np.array] = None
        self.input_shape: Optional[Tuple[int, ...]] = input_shape
        self.n_units: int = int(n_units)
        self.trainable: bool = True
        
        self.W: Optional[np.ndarray] = None
        self.w0: Optional[np.ndarray] = None
        
        self._W_opt = None
        self._w0_opt = None
        
    def initialise(self, optimiser) -> None:
        """
        
        Inititialise params & attach optimisers.
        
        Weight intitialisation:
        
          W = U(-limit, +limit), limit = 1/ sqrt(input_dim)
          
        Params
        
        optimiser: Any
          An object that update(weights: np.ndarray, grad: np.ndarray) -> mp.ndarray
        
        """
        
        if self.input_shape is None:
            raise ValueError(
               "Input_shape must be set" 
            )
        if len(self.input_shape) != 1:
            raise ValueError(
                f"Dense expects a 1D input shape: (in_dim, )"
            )
            
        in_dim = int(self.input_shape[0])
        limit = 1.0 / math.sqrt(in_dim)
        
        self.W = np.random.uniform(-limit, limit, size = (in_dim, self.n_units))
        self.w0 = np.zeros((1, self.n_units), dtype=self.W.dtype)
        
        self._W_opt = copy.deepcopy(optimiser)
        self._w0_opt = copy.deepcopy(optimiser)
        
    def parameters(self) -> int:
        """ Return total param count """
        if self.W is None or self.w0 is None:
            return 0
        return self.W.size + self.w0.size
    
    def forward_pass(self, X:np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute forward pass: X @ W + w0
        
        X: np.ndarray
          Input batch with shape (batch_size, input_dim).
          
        Returns: 
        
        np.ndarray
          Output batch with shape (batch_size, n_units).
        
        """
        
        if self.W is None or self.w0 is None:
            raise RuntimeError("Layer not inititialised")
        
        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Expected input shape (batch_size, {self.W.shape[0]})"
            )
        
        self.layer_input = X
        return X @ self.W + self.w0
    
    def backward_pass(self, accum_grad: np.ndarray) -> np.ndarray:
        
        """
        
        Backpropragate gradients through the layer and update parameters 
        
        accum_grad: np.ndarray
            Upstream gradient dL/dY with shape (batch_size, n_units).
            
        Returns
        
        np.ndarray
            Gradient wrt input, dL/dY with shape (batch_size, n_units).
    
        """ 
        
        if self.W is None or self.w0 is None:
            raise RuntimeError("Layer not inititialised")
        
        if self.layer_input is None:
            raise RuntimeError("No cached input. Perform forward pass")
        
        X = self.layer_input
        
        if accum_grad.ndim != 2 or accum_grad.shape[1] != self.n_units:
            raise ValueError(
                f"Expected upstream gradient shape (batch_size, {self.n_units})"
            )            
            
        grad_W = X.T @ accum_grad
        
        grad_w0 = np.sum(accum_grad, axis = 0, keepdims = True)
        
        grad_input = accum_grad @ self.W.T
        
        if self.trainable:
            if self._W_opt is None or self._w0_opt is None:
                raise RuntimeError("Layer not inititialised")
            
            self.W = self._W_opt.update(self.W, grad_W)
            self.w0 = self._w0_opt.update(self.w0, grad_w0)
            
        self.layer_input = None
        
        return grad_input
    
    def output_shape(self) -> Tuple[int, ...]:
        """
        
        Return the output shape
        
        """    
        return (self.n_units,)
    

class MockOptimiser:
    """ 
    Minimal optimiser: Vanialla SGD woth fixed learning rate 0.01.
    
    """
    
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        
    def update(self, weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return weights - self.lr * grad
    
if __name__ == "__main__":
    
    dense_layer = Dense(n_units = 3, input_shape = (2, ))
    optimiser = MockOptimiser()
    dense_layer.initialise(optimiser)
      
    X = np.array([[1, 2]])
    output = dense_layer.forward_pass(X)
    print("Forward pass output:", output)
    
    accum_grad = np.array([[0.1, 0.2, 0.3]])
    back_output = dense_layer.backward_pass(accum_grad)
    print("Backward pass output:", back_output)
    
    class TestDense(unittest.TestCase):
        
        def setUp(self):
            np.random.seed(42)
            self.in_dim = 4
            self.out_dim = 3
            self.layer = Dense(n_units = self.out_dim, input_shape = (self.in_dim,))
            self.layer.initialise(MockOptimiser(lr = 0.01))
            
        def test_parameters_count(self):
            expected = self.in_dim * self.out_dim + self.out_dim
            self.assertEqual(self.layer.parameters(), expected)
            
        def test_output_shape(self):
            self.assertEqual(self.layer.output_shape(), (self.out_dim,))
            
        def test_forward_shape(self):
            X = np.random.rand(7, self.in_dim)
            Y = self.layer.forward_pass(X)
            self.assertEqual(Y.shape, (7, self.out_dim))
            
        def test_backward_shapes(self):
            X = np.random.rand(3, self.in_dim)
            Y = self.layer.forward_pass(X)
            self.assertEqual(Y.shape, (3, self.out_dim))
            
            dY = np.random.rand(3, self.out_dim)
            dX = self.layer.backward_pass(dY)
            self.assertEqual(dX.shape, (3, self.in_dim))
            
        def test_weights_update(self):
            X = np.random.randn(2, self.in_dim)
            _ = self.layer.forward_pass(X)
            W_before = self.layer.W.copy()
            b_before = self.layer.w0.copy()
            
            dY = np.ones((2, self.out_dim))
            _ = self.layer.backward_pass(dY)
            
            self.assertFalse(np.allclose(W_before, self.layer.W))
            self.assertFalse(np.allclose(b_before, self.layer.w0))
            
        def test_fails_without_init(self):
            uninit = Dense(n_units =2, input_shape = (3,))
            with self.assertRaises(RuntimeError):
                _ = uninit.forward_pass(np.zeros((1, 3)))
                
unittest.main(argv=['first-arg-is-ignored'], exit=False) 
            
    
        
        
