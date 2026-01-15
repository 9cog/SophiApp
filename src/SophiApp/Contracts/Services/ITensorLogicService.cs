// <copyright file="ITensorLogicService.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Contracts.Services;

using SophiApp.Models;

/// <summary>
/// A service for Tensor Logic operations that bridge neural networks and symbolic AI.
/// </summary>
public interface ITensorLogicService
{
    /// <summary>
    /// Creates a new tensor operation.
    /// </summary>
    /// <param name="name">The name of the operation.</param>
    /// <param name="operationType">The type of operation (e.g., "einsum", "matmul").</param>
    /// <param name="inputDimensions">The dimensions of input tensors.</param>
    /// <param name="data">The tensor data as a flattened array.</param>
    /// <returns>A new <see cref="TensorOperation"/> instance.</returns>
    TensorOperation CreateTensorOperation(string name, string operationType, int[] inputDimensions, double[] data);

    /// <summary>
    /// Creates a new logic rule.
    /// </summary>
    /// <param name="name">The name of the rule.</param>
    /// <param name="expression">The rule expression.</param>
    /// <param name="predicate">The predicate name.</param>
    /// <param name="antecedents">The rule antecedents (conditions).</param>
    /// <param name="consequent">The rule consequent (conclusion).</param>
    /// <returns>A new <see cref="LogicRule"/> instance.</returns>
    LogicRule CreateLogicRule(string name, string expression, string predicate, List<string> antecedents, string consequent);

    /// <summary>
    /// Creates a neural-symbolic bridge configuration.
    /// </summary>
    /// <param name="name">The name of the bridge.</param>
    /// <param name="mode">The operation mode (Boolean, Continuous, or Hybrid).</param>
    /// <param name="temperature">The temperature parameter.</param>
    /// <returns>A new <see cref="NeuralSymbolicBridge"/> instance.</returns>
    NeuralSymbolicBridge CreateBridge(string name, OperationMode mode, double temperature);

    /// <summary>
    /// Performs Einstein summation (einsum) operation on tensor data.
    /// </summary>
    /// <param name="equation">The einsum equation (e.g., "ij,jk->ik" for matrix multiplication).</param>
    /// <param name="operands">The input tensor operations.</param>
    /// <returns>A new tensor operation representing the result.</returns>
    TensorOperation PerformEinsum(string equation, params TensorOperation[] operands);

    /// <summary>
    /// Applies a logic rule to derive new facts using tensor operations.
    /// </summary>
    /// <param name="rule">The logic rule to apply.</param>
    /// <param name="facts">The known facts as tensor operations.</param>
    /// <returns>A list of derived tensor operations.</returns>
    List<TensorOperation> ApplyLogicRule(LogicRule rule, List<TensorOperation> facts);

    /// <summary>
    /// Converts a symbolic logic rule to tensor operations.
    /// </summary>
    /// <param name="rule">The logic rule to convert.</param>
    /// <returns>A list of tensor operations representing the rule.</returns>
    List<TensorOperation> RuleToTensor(LogicRule rule);

    /// <summary>
    /// Performs forward inference using tensor logic.
    /// </summary>
    /// <param name="bridge">The neural-symbolic bridge configuration.</param>
    /// <param name="query">The query as a tensor operation.</param>
    /// <returns>The inference result as a tensor operation.</returns>
    TensorOperation ForwardInference(NeuralSymbolicBridge bridge, TensorOperation query);

    /// <summary>
    /// Computes the gradient for a tensor operation (for learning).
    /// </summary>
    /// <param name="operation">The tensor operation.</param>
    /// <param name="target">The target values.</param>
    /// <returns>The gradient as a tensor operation.</returns>
    TensorOperation ComputeGradient(TensorOperation operation, double[] target);

    /// <summary>
    /// Validates a tensor operation.
    /// </summary>
    /// <param name="operation">The tensor operation to validate.</param>
    /// <returns>True if the operation is valid; otherwise, false.</returns>
    bool ValidateTensorOperation(TensorOperation operation);

    /// <summary>
    /// Validates a logic rule.
    /// </summary>
    /// <param name="rule">The logic rule to validate.</param>
    /// <returns>True if the rule is valid; otherwise, false.</returns>
    bool ValidateLogicRule(LogicRule rule);
}
