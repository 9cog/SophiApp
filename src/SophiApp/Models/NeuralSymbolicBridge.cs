// <copyright file="NeuralSymbolicBridge.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Models;

/// <summary>
/// Represents the bridge between neural and symbolic representations in Tensor Logic.
/// </summary>
public class NeuralSymbolicBridge
{
    /// <summary>
    /// Gets or sets the unique identifier for the bridge.
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the bridge configuration.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the temperature parameter (0.0 = strict logic, higher = more neural/probabilistic).
    /// </summary>
    public double Temperature { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the mode of operation.
    /// </summary>
    public OperationMode Mode { get; set; } = OperationMode.Boolean;

    /// <summary>
    /// Gets or sets the associated tensor operations.
    /// </summary>
    public List<TensorOperation> TensorOperations { get; set; } = new List<TensorOperation>();

    /// <summary>
    /// Gets or sets the associated logic rules.
    /// </summary>
    public List<LogicRule> LogicRules { get; set; } = new List<LogicRule>();

    /// <summary>
    /// Gets or sets the embedding dimension for continuous representations.
    /// </summary>
    public int EmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets a value indicating whether gradient computation is enabled.
    /// </summary>
    public bool EnableGradients { get; set; } = false;

    /// <summary>
    /// Gets or sets the timestamp when the bridge was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Defines the operation mode for neural-symbolic integration.
/// </summary>
public enum OperationMode
{
    /// <summary>
    /// Boolean/discrete mode for exact logical reasoning.
    /// </summary>
    Boolean,

    /// <summary>
    /// Continuous/differentiable mode for learning and probabilistic inference.
    /// </summary>
    Continuous,

    /// <summary>
    /// Hybrid mode combining both approaches.
    /// </summary>
    Hybrid
}
