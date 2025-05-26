import numpy as np

class SimpleRNN:
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(1)
        
        self.W_input_hidden = rng.randn(hidden_dim, input_dim) * 0.001
        self.W_hidden_hidden = rng.randn(hidden_dim, hidden_dim) * 0.001
        self.W_hidden_output = rng.randn(output_dim, hidden_dim) * 0.001
        
        self.b_hidden = np.zeros((hidden_dim, 1))
        self.b_output = np.zeros((output_dim, 1))
        
    def forward(self, inputs: np.array):
        
        T = len(inputs)
        hidden_states = np.zeros((T+1, self.hidden_dim))
        outputs = np.zeros((T, self.W_hidden_output.shape[0]))
        
        for t in range(T):
            X_t = inputs[t].reshape(-1, 1)
            h_prev = hidden_states[t].reshape(-1, 1)
            
            z = (self.W_input_hidden @ X_t +
                 self.W_hidden_hidden @ h_prev +
                 self.b_hidden)
            h_new = np.tanh(z)
            hidden_states[t+1] = h_new.ravel()
            
            y_t = self.W_hidden_output @ h_new + self.b_output
            outputs[t] = y_t.ravel()
        
        return outputs, hidden_states
    
    def train_step(self, inputs: np.array, targets: np.array, lr: float = 0.001):
        
        outputs, hidden_states = self.forward(inputs)
        T = len(inputs)
        
        diff = outputs - targets 
        loss = 0.5 * np.sum(diff**2)
        d_output = diff
        
        dW_ho = np.zeros_like(self.W_hidden_output)
        db_o = np.zeros_like(self.b_output)
        dW_ih = np.zeros_like(self.W_input_hidden)
        dW_hh = np.zeros_like(self.W_hidden_hidden)
        db_h = np.zeros_like(self.b_hidden)
        dh_next = np.zeros((self.hidden_dim, 1))
        
        for t in reversed(range(T)):
            dy = d_output[t].reshape(-1, 1)
            
            
            dW_ho += dy @ hidden_states[t+1].reshape(1, -1)
            db_o += dy
            
            dh = self.W_hidden_output.T @ dy + dh_next
            
            dz = (1 - hidden_states[t+1].reshape(-1, 1)**2) * dh
            
            x_t = inputs[t].reshape(1, -1)
            h_prev = hidden_states[t].reshape(1, -1)
            dW_ih += dz @ x_t
            dW_hh += dz @ h_prev
            db_h += dz
            
            dh_next = self.W_hidden_hidden.T @ dz
            
        for grad in (dW_ho, db_o, dW_ih, dW_hh, db_h):
            np.clip(grad, -5, 5, out=grad)
            
        self.W_hidden_output -= lr * dW_ho
        self.b_output        -= lr * db_o
        self.W_input_hidden  -= lr * dW_ih
        self.W_hidden_hidden -= lr * dW_hh
        self.b_hidden        -= lr * db_h
        
        return loss
    
if __name__ == "__main__":
    
    seq = np.arange(1, 6).reshape(-1, 1) / 5.0
    target = np.arange(2, 7).reshape(-1, 1) / 5.0
    
    model = SimpleRNN(input_dim = 1, hidden_dim = 4, output_dim = 1)
    for epoch in range(200):
        loss = model.train_step(seq, target, lr = 0.01)
        
    print(f"Final loss: {loss:.6f}")
    preds, _ = model.forward(seq)
    print("Predictions:", np.round(preds.ravel(), 3))       
            
        
        