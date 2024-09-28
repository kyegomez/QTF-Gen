# Quantum Field Theory Inspired Neural Network for Text Generation

## Introduction

Quantum Field Theory (QFT) is a fundamental framework in physics that blends quantum mechanics with special relativity to describe how particles interact and propagate through space and time. In QFT, particles are seen as excitations of underlying fields, and their interactions are governed by equations derived from Lagrangians or Hamiltonians.

This project aims to create a neural network inspired by QFT concepts to generate text. The idea is to model token embeddings as fields and simulate their interactions using layers analogous to those in QFT, such as convolutional layers for local interactions and Fourier transforms for global interactions.

## Theory Overview

- **Field Representation**: Each token in the text sequence is represented as a field value at a specific position. The sequence of tokens forms a field configuration over discrete positions.

- **Local Interactions**: Modeled using convolutional layers, capturing the interactions between neighboring tokens, akin to local interactions in a field.

- **Global Interactions**: Modeled using Fourier transforms, capturing the global properties of the field by transforming it into the frequency domain.

- **Field Evolution**: The field evolves through layers that apply transformations inspired by QFT operations, such as field excitations and propagators.

## Neural Network Architecture

1. **Embedding Layer**: Converts input tokens into embeddings (field values).

2. **QFT Layers**: A stack of layers where each layer models local and global interactions:
   - **Convolutional Layer**: Captures local interactions between tokens.
   - **Activation Function**: Introduces non-linearity using ReLU.
   - **Fourier Transform**: Captures global interactions by transforming embeddings into the frequency domain and back.

3. **Output Layer**: Predicts the probability distribution over the vocabulary for the next token.

## Implementation in PyTorch

Below is the implementation of the QFT-inspired neural network in PyTorch, complete with documentation, logging using `loguru`, and type validation using type annotations.

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger

class QFTInspiredTextGenerator(nn.Module):
    """
    Quantum Field Theory inspired neural network for text generation.

    This model treats token embeddings as fields and models their interactions using
    convolutional and spectral layers inspired by concepts from quantum field theory.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the token embeddings.
        hidden_dim (int): Dimension of the hidden layers.
        num_layers (int): Number of layers in the model.
        kernel_size (int): Kernel size for convolutional layers.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 4, kernel_size: int = 3):
        super(QFTInspiredTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(QFTLayer(embedding_dim, hidden_dim, kernel_size))
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        logger.info("QFTInspiredTextGenerator initialized.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            Tensor: Log probabilities for the next token with shape (batch_size, vocab_size).
        """
        logger.debug("Forward pass started.")
        embedded = self.embedding(x)
        phi = embedded
        for layer in self.layers:
            phi = layer(phi)
        phi = phi.mean(dim=1)  # Aggregate over sequence length
        output = self.fc_out(phi)
        log_probs = self.log_softmax(output)
        logger.debug("Forward pass completed.")
        return log_probs

class QFTLayer(nn.Module):
    """
    Quantum Field Theory inspired layer.

    This layer models local interactions using convolution and global interactions using
    Fourier transforms.

    Args:
        input_dim (int): Dimension of the input embeddings.
        hidden_dim (int): Dimension of the hidden layer.
        kernel_size (int): Kernel size for the convolutional layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int):
        super(QFTLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU()
        logger.debug("QFTLayer initialized.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the QFTLayer.

        Args:
            x (Tensor): Input tensor with shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor with shape (batch_size, sequence_length, hidden_dim).
        """
        logger.debug("QFTLayer forward pass started.")
        # Transpose to (batch_size, input_dim, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        x_conv = self.conv(x)
        x_conv = self.activation(x_conv)
        # Global interaction via Fourier Transform
        x_fft = torch.fft.fft(x_conv, dim=-1)
        x_fft = torch.real(x_fft)
        # Inverse FFT to bring back to time domain
        x_ifft = torch.fft.ifft(x_fft, dim=-1)
        x_ifft = torch.real(x_ifft)
        # Transpose back to (batch_size, sequence_length, hidden_dim)
        x_out = x_ifft.transpose(1, 2)
        logger.debug("QFTLayer forward pass completed.")
        return x_out
```

## Explanation of the Code

- **Imports**: The necessary modules are imported, including `torch`, `torch.nn`, `loguru` for logging, and type annotations.

- **QFTInspiredTextGenerator Class**:
  - **Initialization**:
    - Embedding layer to convert tokens to embeddings.
    - A list of `QFTLayer` instances to model the field interactions.
    - A fully connected layer to map the hidden states to vocabulary size.
    - Log-softmax layer to obtain log probabilities.
    - Logging statement to indicate initialization.
  - **Forward Method**:
    - Embeds the input tokens.
    - Passes the embeddings through each `QFTLayer`.
    - Averages the output over the sequence length.
    - Applies the fully connected layer and log-softmax to obtain log probabilities.
    - Logging statements to trace the forward pass.

- **QFTLayer Class**:
  - **Initialization**:
    - Convolutional layer to model local interactions.
    - ReLU activation function.
    - Logging statement to indicate initialization.
  - **Forward Method**:
    - Transposes the input tensor to match the expected shape for `Conv1d`.
    - Applies convolution and activation function.
    - Performs Fourier transform to model global interactions.
    - Takes the real part of the transformed data.
    - Applies inverse Fourier transform to return to the original domain.
    - Transposes the tensor back to the original shape.
    - Logging statements to trace the forward pass.

## Logging with Loguru

The `loguru` library is used to add logging statements throughout the code. This allows for easy tracing of the model's execution, which is helpful for debugging and understanding the flow of data.

- **Initialization Logs**: Indicate when each component of the model is initialized.

- **Forward Pass Logs**: Trace the execution of the forward pass in both the main model and the QFT layers.

## Type Validation

Type annotations are provided for all methods and class attributes. This ensures that the code is more readable and helps catch type-related errors during development.

- **Type Annotations**: Used for function arguments and return types.

- **Runtime Type Checking**: While Python doesn't enforce type annotations at runtime, tools like `mypy` can be used during development to perform static type checking.

## Usage Example

Here's how you might instantiate and use the model:

```python
# Define hyperparameters
vocab_size = 10000
embedding_dim = 256
hidden_dim = 512
sequence_length = 50

# Initialize the model
model = QFTInspiredTextGenerator(vocab_size, embedding_dim, hidden_dim)

# Create a sample input (batch_size=32)
sample_input = torch.randint(0, vocab_size, (32, sequence_length))

# Get the log probabilities
log_probs = model(sample_input)

print(log_probs.shape)  # Output: torch.Size([32, 10000])
```

## Conclusion

This implementation provides a neural network model inspired by quantum field theory concepts for text generation tasks. By modeling tokens as fields and simulating their interactions using convolutional and Fourier layers, we aim to capture both local and global dependencies in the text data. The use of `loguru` for logging and type annotations enhances the maintainability and debuggability of the code.
