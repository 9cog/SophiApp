// <copyright file="TensorOperation.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Models;

/// <summary>
/// Represents a tensor operation in the Tensor Logic system.
/// </summary>
public class TensorOperation
{
    /// <summary>
    /// Gets or sets the unique identifier for the tensor operation.
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the tensor operation.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the operation type (e.g., "einsum", "matmul", "add").
    /// </summary>
    public string OperationType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the input tensor dimensions.
    /// </summary>
    public int[] InputDimensions { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the output tensor dimensions.
    /// </summary>
    public int[] OutputDimensions { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the tensor data as a flattened array.
    /// </summary>
    public double[] Data { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the timestamp when the operation was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets a value indicating whether this operation is differentiable (supports backpropagation).
    /// </summary>
    public bool IsDifferentiable { get; set; } = true;

    /// <summary>
    /// Gets or sets the operation parameters as key-value pairs.
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
}
