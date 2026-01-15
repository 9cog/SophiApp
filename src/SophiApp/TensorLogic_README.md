# Tensor Logic Implementation for SophiApp

## Overview

This implementation provides a foundational framework for Tensor Logic - a unified approach to combining neural networks (deep learning) with symbolic AI (logical reasoning). This is based on the concepts from [tensor-logic.org](https://tensor-logic.org/) and related research.

## What is Tensor Logic?

Tensor Logic is a programming paradigm that unifies:
- **Neural Networks**: Scalable, gradient-based learning with continuous representations
- **Symbolic AI**: Transparent, reliable logical reasoning with discrete rules

The key insight is that both neural networks and logic programs can be expressed as **tensor operations**, primarily using Einstein summation (einsum) notation.

## Features Implemented

### 1. Core Models

#### TensorOperation
Represents tensor operations in the system:
- Supports various operation types (einsum, matmul, add, etc.)
- Stores multi-dimensional tensor data
- Tracks input/output dimensions
- Supports differentiation for gradient-based learning

#### LogicRule
Represents symbolic logic rules:
- Expresses rules in logical notation (e.g., "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)")
- Supports both hard (exact) and soft (probabilistic) rules
- Includes confidence scores for uncertainty handling
- Stores rule metadata

#### NeuralSymbolicBridge
Bridges neural and symbolic representations:
- **Boolean Mode**: Exact logical reasoning without approximation
- **Continuous Mode**: Soft, differentiable probabilistic inference
- **Hybrid Mode**: Combines both approaches with temperature control

### 2. TensorLogicService

The main service provides:

#### Tensor Operations
- `CreateTensorOperation`: Creates new tensor operations
- `PerformEinsum`: Implements Einstein summation for tensor manipulation
- Matrix multiplication and other tensor operations

#### Logic Operations
- `CreateLogicRule`: Defines new logical rules
- `ApplyLogicRule`: Applies rules to derive new facts
- `RuleToTensor`: Converts symbolic rules to tensor representations

#### Neural-Symbolic Integration
- `CreateBridge`: Configures the neural-symbolic bridge
- `ForwardInference`: Performs inference in Boolean, Continuous, or Hybrid modes
- `ComputeGradient`: Calculates gradients for learning

#### Validation
- `ValidateTensorOperation`: Ensures tensor operations are well-formed
- `ValidateLogicRule`: Validates logical rule correctness

## Usage Examples

### Example 1: Creating Tensor Operations

```csharp
var service = new TensorLogicService();

// Create a 2x2 matrix
var tensorA = service.CreateTensorOperation(
    name: "MatrixA",
    operationType: "matrix",
    inputDimensions: new[] { 2, 2 },
    data: new[] { 1.0, 2.0, 3.0, 4.0 }
);

// Create another 2x2 matrix
var tensorB = service.CreateTensorOperation(
    name: "MatrixB",
    operationType: "matrix",
    inputDimensions: new[] { 2, 2 },
    data: new[] { 5.0, 6.0, 7.0, 8.0 }
);

// Perform matrix multiplication using einsum
var result = service.PerformEinsum("ij,jk->ik", tensorA, tensorB);
// Result: [[19,22],[43,50]]
```

### Example 2: Creating Logic Rules

```csharp
// Define a grandparent rule
var rule = service.CreateLogicRule(
    name: "Grandparent Rule",
    expression: "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)",
    predicate: "grandparent",
    antecedents: new List<string> { "parent(X,Y)", "parent(Y,Z)" },
    consequent: "grandparent(X,Z)"
);

// Validate the rule
bool isValid = service.ValidateLogicRule(rule);
```

### Example 3: Neural-Symbolic Inference

```csharp
// Create a bridge for boolean (exact) reasoning
var booleanBridge = service.CreateBridge(
    name: "Exact Reasoning",
    mode: OperationMode.Boolean,
    temperature: 0.0
);

// Create a query
var query = service.CreateTensorOperation(
    name: "Query",
    operationType: "predicate",
    inputDimensions: new[] { 4 },
    data: new[] { 0.3, 0.7, 0.2, 0.9 }
);

// Perform boolean inference (threshold at 0.5)
var result = service.ForwardInference(booleanBridge, query);
// Result: [0.0, 1.0, 0.0, 1.0]

// Create a bridge for continuous (soft) reasoning
var continuousBridge = service.CreateBridge(
    name: "Probabilistic Reasoning",
    mode: OperationMode.Continuous,
    temperature: 1.0
);

// Perform continuous inference (softmax normalization)
var softResult = service.ForwardInference(continuousBridge, query);
// Result: probability distribution summing to 1.0
```

### Example 4: Hybrid Mode with Temperature Control

```csharp
// Create a hybrid bridge
var hybridBridge = service.CreateBridge(
    name: "Hybrid Reasoning",
    mode: OperationMode.Hybrid,
    temperature: 0.5  // 0.0 = pure Boolean, 1.0 = pure Continuous
);

var query = service.CreateTensorOperation(
    name: "Query",
    operationType: "predicate",
    inputDimensions: new[] { 3 },
    data: new[] { 0.2, 0.6, 0.8 }
);

// Perform hybrid inference (blends Boolean and Continuous)
var result = service.ForwardInference(hybridBridge, query);
```

### Example 5: Learning with Gradients

```csharp
// Create a differentiable operation
var operation = service.CreateTensorOperation(
    name: "Learned Parameters",
    operationType: "linear",
    inputDimensions: new[] { 3 },
    data: new[] { 1.0, 2.0, 3.0 }
);

// Define target values
var target = new[] { 0.5, 1.5, 2.5 };

// Compute gradient for learning
var gradient = service.ComputeGradient(operation, target);
// Result: [0.5, 0.5, 0.5] (difference between output and target)
```

## Key Concepts

### Temperature Parameter
Controls the balance between strict logic and creative/probabilistic reasoning:
- **Temperature = 0.0**: Strict Boolean logic (exact, provable)
- **Temperature > 0.0**: Soft probabilistic reasoning (learning-friendly)
- **Temperature = 1.0**: Standard softmax temperature
- **Temperature > 1.0**: More uniform (creative) distributions

### Operation Modes

1. **Boolean Mode**
   - Exact logical inference
   - No hallucinations
   - Threshold-based decisions (values > 0.5 → 1.0, else → 0.0)
   - Use for: Provable facts, strict requirements

2. **Continuous Mode**
   - Probabilistic inference
   - Supports learning via gradients
   - Softmax normalization
   - Use for: Uncertainty, learning, pattern matching

3. **Hybrid Mode**
   - Combines both approaches
   - Temperature-weighted blending
   - Use for: Balanced reasoning with some flexibility

### Einstein Summation (Einsum)
Einsum is a compact notation for tensor operations:
- `"ij,jk->ik"`: Matrix multiplication
- `"ii->i"`: Diagonal extraction
- `"ij->ji"`: Matrix transpose
- Supports complex tensor manipulations in a single operation

## Testing

The implementation includes comprehensive unit tests covering:
- All model classes (TensorOperation, LogicRule, NeuralSymbolicBridge)
- All service methods
- Edge cases and error conditions
- Validation logic

Run tests with:
```bash
dotnet test src/SophiApp.Tests/SophiApp.Tests.csproj
```

## Architecture Integration

This Tensor Logic implementation is designed as a service within the SophiApp architecture:
- Follows the MVVM pattern used throughout SophiApp
- Implements contract interfaces (ITensorLogicService)
- Can be injected via dependency injection
- Integrates with existing SophiApp services

## Future Enhancements

Potential areas for expansion:
1. **Predicate Invention**: Automatic discovery of hidden relations via tensor factorization
2. **Advanced Inference**: Implement forward/backward chaining algorithms
3. **GPU Acceleration**: Leverage GPU for large-scale tensor operations
4. **Rule Learning**: Learn logic rules from data using gradient descent
5. **Integration with ML.NET**: Connect to .NET machine learning frameworks
6. **Visualization**: Add UI components to visualize tensor operations and logic rules

## References

- [Tensor Logic Website](https://tensor-logic.org/)
- [Tensor Logic Paper (arXiv)](https://arxiv.org/abs/2510.12269)
- [Pedro Domingos - Tensor Logic Slides](https://homes.cs.washington.edu/~pedrod/tls.pdf)
- [Machine Learning Street Talk Interview](https://www.youtube.com/watch?v=4APMGvicmxY)

## License

Copyright (c) Team Sophia. All rights reserved.

This implementation is part of SophiApp and follows the same license terms.
