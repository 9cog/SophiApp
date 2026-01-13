// <copyright file="TensorLogicService.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Services;

using SophiApp.Contracts.Services;
using SophiApp.Models;
using System;
using System.Collections.Generic;
using System.Linq;

/// <inheritdoc/>
public class TensorLogicService : ITensorLogicService
{
    /// <inheritdoc/>
    public TensorOperation CreateTensorOperation(string name, string operationType, int[] inputDimensions, double[] data)
    {
        var operation = new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            OperationType = operationType,
            InputDimensions = inputDimensions,
            Data = data,
            CreatedAt = DateTime.UtcNow
        };

        // Calculate output dimensions based on operation type
        operation.OutputDimensions = CalculateOutputDimensions(operationType, inputDimensions);

        return operation;
    }

    /// <inheritdoc/>
    public LogicRule CreateLogicRule(string name, string expression, string predicate, List<string> antecedents, string consequent)
    {
        return new LogicRule
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            Expression = expression,
            Predicate = predicate,
            Antecedents = antecedents,
            Consequent = consequent,
            CreatedAt = DateTime.UtcNow
        };
    }

    /// <inheritdoc/>
    public NeuralSymbolicBridge CreateBridge(string name, OperationMode mode, double temperature)
    {
        return new NeuralSymbolicBridge
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            Mode = mode,
            Temperature = temperature,
            EnableGradients = mode != OperationMode.Boolean,
            CreatedAt = DateTime.UtcNow
        };
    }

    /// <inheritdoc/>
    public TensorOperation PerformEinsum(string equation, params TensorOperation[] operands)
    {
        if (operands == null || operands.Length == 0)
        {
            throw new ArgumentException("At least one operand is required for einsum operation.", nameof(operands));
        }

        // Parse the einsum equation (e.g., "ij,jk->ik")
        var parts = equation.Split(new[] { "->" }, StringSplitOptions.None);
        if (parts.Length != 2)
        {
            throw new ArgumentException($"Invalid einsum equation: {equation}", nameof(equation));
        }

        var inputSpecs = parts[0].Split(',');
        var outputSpec = parts[1];

        // For demonstration, we'll implement matrix multiplication (ij,jk->ik)
        if (equation == "ij,jk->ik" && operands.Length == 2)
        {
            return PerformMatrixMultiplication(operands[0], operands[1]);
        }

        // For other cases, return a placeholder result
        var resultData = new double[operands[0].Data.Length];
        Array.Copy(operands[0].Data, resultData, resultData.Length);

        return new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"einsum_{equation}",
            OperationType = "einsum",
            InputDimensions = operands[0].InputDimensions,
            OutputDimensions = operands[0].OutputDimensions,
            Data = resultData,
            CreatedAt = DateTime.UtcNow,
            Parameters = new Dictionary<string, object> { { "equation", equation } }
        };
    }

    /// <inheritdoc/>
    public List<TensorOperation> ApplyLogicRule(LogicRule rule, List<TensorOperation> facts)
    {
        var derivedFacts = new List<TensorOperation>();

        if (facts == null || facts.Count == 0)
        {
            return derivedFacts;
        }

        // Convert the logic rule to tensor operations
        var ruleTensors = RuleToTensor(rule);

        // For each fact, apply the rule using tensor operations
        foreach (var fact in facts)
        {
            // Simplified rule application: if the fact matches an antecedent,
            // derive the consequent using tensor multiplication
            if (ruleTensors.Any() && fact.Name.Contains(rule.Predicate))
            {
                var derived = new TensorOperation
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = $"{rule.Consequent}_derived",
                    OperationType = "rule_application",
                    InputDimensions = fact.InputDimensions,
                    OutputDimensions = fact.OutputDimensions,
                    Data = ApplySoftmaxIfNeeded(fact.Data, rule.Confidence),
                    CreatedAt = DateTime.UtcNow,
                    Parameters = new Dictionary<string, object>
                    {
                        { "source_rule", rule.Id },
                        { "confidence", rule.Confidence }
                    }
                };
                derivedFacts.Add(derived);
            }
        }

        return derivedFacts;
    }

    /// <inheritdoc/>
    public List<TensorOperation> RuleToTensor(LogicRule rule)
    {
        var tensorOperations = new List<TensorOperation>();

        // For each antecedent, create a tensor representation
        foreach (var antecedent in rule.Antecedents)
        {
            var tensor = new TensorOperation
            {
                Id = Guid.NewGuid().ToString(),
                Name = $"{antecedent}_tensor",
                OperationType = "predicate",
                InputDimensions = new[] { 10, 10 }, // Simplified: assume 10x10 relation matrices
                OutputDimensions = new[] { 10, 10 },
                Data = InitializeIdentityMatrix(10),
                CreatedAt = DateTime.UtcNow,
                Parameters = new Dictionary<string, object> { { "predicate", antecedent } }
            };
            tensorOperations.Add(tensor);
        }

        // Create a tensor for the consequent
        var consequentTensor = new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"{rule.Consequent}_tensor",
            OperationType = "predicate",
            InputDimensions = new[] { 10, 10 },
            OutputDimensions = new[] { 10, 10 },
            Data = InitializeZeroMatrix(10),
            CreatedAt = DateTime.UtcNow,
            Parameters = new Dictionary<string, object> { { "predicate", rule.Consequent } }
        };
        tensorOperations.Add(consequentTensor);

        return tensorOperations;
    }

    /// <inheritdoc/>
    public TensorOperation ForwardInference(NeuralSymbolicBridge bridge, TensorOperation query)
    {
        if (bridge.Mode == OperationMode.Boolean)
        {
            // Perform exact logical inference
            return PerformBooleanInference(bridge, query);
        }
        else if (bridge.Mode == OperationMode.Continuous)
        {
            // Perform soft/probabilistic inference
            return PerformContinuousInference(bridge, query);
        }
        else
        {
            // Hybrid mode: combine both approaches
            var booleanResult = PerformBooleanInference(bridge, query);
            var continuousResult = PerformContinuousInference(bridge, query);
            return CombineResults(booleanResult, continuousResult, bridge.Temperature);
        }
    }

    /// <inheritdoc/>
    public TensorOperation ComputeGradient(TensorOperation operation, double[] target)
    {
        if (!operation.IsDifferentiable)
        {
            throw new InvalidOperationException("Operation is not differentiable.");
        }

        // Compute simple gradient: difference between output and target
        var gradient = new double[operation.Data.Length];
        for (int i = 0; i < operation.Data.Length && i < target.Length; i++)
        {
            gradient[i] = operation.Data[i] - target[i];
        }

        return new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"{operation.Name}_gradient",
            OperationType = "gradient",
            InputDimensions = operation.OutputDimensions,
            OutputDimensions = operation.OutputDimensions,
            Data = gradient,
            CreatedAt = DateTime.UtcNow,
            IsDifferentiable = false
        };
    }

    /// <inheritdoc/>
    public bool ValidateTensorOperation(TensorOperation operation)
    {
        if (operation == null)
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(operation.Name))
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(operation.OperationType))
        {
            return false;
        }

        if (operation.InputDimensions == null || operation.InputDimensions.Length == 0)
        {
            return false;
        }

        if (operation.Data == null)
        {
            return false;
        }

        // Validate that data size matches dimensions
        var expectedSize = operation.InputDimensions.Aggregate(1, (a, b) => a * b);
        if (operation.Data.Length != expectedSize)
        {
            return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public bool ValidateLogicRule(LogicRule rule)
    {
        if (rule == null)
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(rule.Name))
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(rule.Expression))
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(rule.Predicate))
        {
            return false;
        }

        if (rule.Antecedents == null || rule.Antecedents.Count == 0)
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(rule.Consequent))
        {
            return false;
        }

        if (rule.Confidence < 0.0 || rule.Confidence > 1.0)
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Calculates output dimensions based on operation type and input dimensions.
    /// </summary>
    /// <param name="operationType">The type of operation.</param>
    /// <param name="inputDimensions">The input dimensions.</param>
    /// <returns>The calculated output dimensions.</returns>
    private static int[] CalculateOutputDimensions(string operationType, int[] inputDimensions)
    {
        return operationType switch
        {
            "matmul" => new[] { inputDimensions[0], inputDimensions.Length > 2 ? inputDimensions[2] : inputDimensions[0] },
            "transpose" => inputDimensions.Reverse().ToArray(),
            "einsum" => inputDimensions, // Simplified: keep same dimensions
            _ => inputDimensions
        };
    }

    /// <summary>
    /// Performs matrix multiplication on two tensor operations.
    /// </summary>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <returns>The result of matrix multiplication.</returns>
    private static TensorOperation PerformMatrixMultiplication(TensorOperation a, TensorOperation b)
    {
        if (a.InputDimensions.Length < 2 || b.InputDimensions.Length < 2)
        {
            throw new ArgumentException("Matrix multiplication requires 2D tensors.");
        }

        int m = a.InputDimensions[0];
        int n = a.InputDimensions[1];
        int p = b.InputDimensions[1];

        if (n != b.InputDimensions[0])
        {
            throw new ArgumentException("Inner dimensions must match for matrix multiplication.");
        }

        var result = new double[m * p];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a.Data[i * n + k] * b.Data[k * p + j];
                }
                result[i * p + j] = sum;
            }
        }

        return new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"{a.Name}_x_{b.Name}",
            OperationType = "matmul",
            InputDimensions = new[] { m, p },
            OutputDimensions = new[] { m, p },
            Data = result,
            CreatedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Applies softmax transformation if confidence is less than 1.0.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="confidence">The confidence level.</param>
    /// <returns>The transformed data.</returns>
    private static double[] ApplySoftmaxIfNeeded(double[] data, double confidence)
    {
        if (confidence >= 1.0)
        {
            return (double[])data.Clone();
        }

        var result = new double[data.Length];
        var max = data.Max();
        var expSum = data.Sum(x => Math.Exp(x - max));

        for (int i = 0; i < data.Length; i++)
        {
            result[i] = Math.Exp(data[i] - max) / expSum * confidence;
        }

        return result;
    }

    /// <summary>
    /// Initializes an identity matrix.
    /// </summary>
    /// <param name="size">The size of the matrix.</param>
    /// <returns>A flattened identity matrix.</returns>
    private static double[] InitializeIdentityMatrix(int size)
    {
        var matrix = new double[size * size];
        for (int i = 0; i < size; i++)
        {
            matrix[i * size + i] = 1.0;
        }
        return matrix;
    }

    /// <summary>
    /// Initializes a zero matrix.
    /// </summary>
    /// <param name="size">The size of the matrix.</param>
    /// <returns>A flattened zero matrix.</returns>
    private static double[] InitializeZeroMatrix(int size)
    {
        return new double[size * size];
    }

    /// <summary>
    /// Performs boolean (exact) inference.
    /// </summary>
    /// <param name="bridge">The neural-symbolic bridge.</param>
    /// <param name="query">The query tensor.</param>
    /// <returns>The inference result.</returns>
    private static TensorOperation PerformBooleanInference(NeuralSymbolicBridge bridge, TensorOperation query)
    {
        // Apply boolean logic: threshold at 0.5
        var result = new double[query.Data.Length];
        for (int i = 0; i < query.Data.Length; i++)
        {
            result[i] = query.Data[i] > 0.5 ? 1.0 : 0.0;
        }

        return new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"{query.Name}_boolean_inference",
            OperationType = "boolean_inference",
            InputDimensions = query.InputDimensions,
            OutputDimensions = query.OutputDimensions,
            Data = result,
            CreatedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Performs continuous (soft/probabilistic) inference.
    /// </summary>
    /// <param name="bridge">The neural-symbolic bridge.</param>
    /// <param name="query">The query tensor.</param>
    /// <returns>The inference result.</returns>
    private static TensorOperation PerformContinuousInference(NeuralSymbolicBridge bridge, TensorOperation query)
    {
        // Apply softmax with temperature
        var temperature = Math.Max(bridge.Temperature, 0.01); // Avoid division by zero
        var result = new double[query.Data.Length];
        var max = query.Data.Max();
        var expSum = query.Data.Sum(x => Math.Exp((x - max) / temperature));

        for (int i = 0; i < query.Data.Length; i++)
        {
            result[i] = Math.Exp((query.Data[i] - max) / temperature) / expSum;
        }

        return new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"{query.Name}_continuous_inference",
            OperationType = "continuous_inference",
            InputDimensions = query.InputDimensions,
            OutputDimensions = query.OutputDimensions,
            Data = result,
            CreatedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Combines boolean and continuous inference results.
    /// </summary>
    /// <param name="booleanResult">The boolean inference result.</param>
    /// <param name="continuousResult">The continuous inference result.</param>
    /// <param name="temperature">The temperature parameter for blending.</param>
    /// <returns>The combined result.</returns>
    private static TensorOperation CombineResults(TensorOperation booleanResult, TensorOperation continuousResult, double temperature)
    {
        var weight = Math.Min(temperature, 1.0); // Use temperature to weight the combination
        var result = new double[booleanResult.Data.Length];

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = (1.0 - weight) * booleanResult.Data[i] + weight * continuousResult.Data[i];
        }

        return new TensorOperation
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"{booleanResult.Name}_hybrid",
            OperationType = "hybrid_inference",
            InputDimensions = booleanResult.InputDimensions,
            OutputDimensions = booleanResult.OutputDimensions,
            Data = result,
            CreatedAt = DateTime.UtcNow
        };
    }
}
