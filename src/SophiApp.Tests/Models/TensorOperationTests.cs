// <copyright file="TensorOperationTests.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Tests.Models;

using SophiApp.Models;
using Xunit;

/// <summary>
/// Unit tests for the TensorOperation model.
/// </summary>
public class TensorOperationTests
{
    [Fact]
    public void Constructor_ShouldInitializeDefaultValues()
    {
        // Arrange & Act
        var operation = new TensorOperation();

        // Assert
        Assert.NotNull(operation);
        Assert.Equal(string.Empty, operation.Id);
        Assert.Equal(string.Empty, operation.Name);
        Assert.Equal(string.Empty, operation.OperationType);
        Assert.Empty(operation.InputDimensions);
        Assert.Empty(operation.OutputDimensions);
        Assert.Empty(operation.Data);
        Assert.True(operation.IsDifferentiable);
        Assert.NotNull(operation.Parameters);
        Assert.Empty(operation.Parameters);
    }

    [Fact]
    public void Properties_ShouldSetAndGetCorrectly()
    {
        // Arrange
        var operation = new TensorOperation
        {
            Id = "test-id",
            Name = "test-operation",
            OperationType = "matmul",
            InputDimensions = new[] { 2, 3 },
            OutputDimensions = new[] { 2, 2 },
            Data = new[] { 1.0, 2.0, 3.0, 4.0 },
            IsDifferentiable = false
        };

        // Act & Assert
        Assert.Equal("test-id", operation.Id);
        Assert.Equal("test-operation", operation.Name);
        Assert.Equal("matmul", operation.OperationType);
        Assert.Equal(new[] { 2, 3 }, operation.InputDimensions);
        Assert.Equal(new[] { 2, 2 }, operation.OutputDimensions);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, operation.Data);
        Assert.False(operation.IsDifferentiable);
    }

    [Fact]
    public void CreatedAt_ShouldBeSetAutomatically()
    {
        // Arrange
        var beforeCreation = DateTime.UtcNow;

        // Act
        var operation = new TensorOperation();
        var afterCreation = DateTime.UtcNow;

        // Assert
        Assert.True(operation.CreatedAt >= beforeCreation);
        Assert.True(operation.CreatedAt <= afterCreation);
    }

    [Fact]
    public void Parameters_ShouldStoreKeyValuePairs()
    {
        // Arrange
        var operation = new TensorOperation();

        // Act
        operation.Parameters["equation"] = "ij,jk->ik";
        operation.Parameters["learning_rate"] = 0.01;

        // Assert
        Assert.Equal(2, operation.Parameters.Count);
        Assert.Equal("ij,jk->ik", operation.Parameters["equation"]);
        Assert.Equal(0.01, operation.Parameters["learning_rate"]);
    }

    [Fact]
    public void Data_ShouldHandleLargeArrays()
    {
        // Arrange
        var largeArray = new double[1000];
        for (int i = 0; i < largeArray.Length; i++)
        {
            largeArray[i] = i * 0.1;
        }

        // Act
        var operation = new TensorOperation { Data = largeArray };

        // Assert
        Assert.Equal(1000, operation.Data.Length);
        Assert.Equal(0.0, operation.Data[0]);
        Assert.Equal(99.9, operation.Data[999], precision: 1);
    }

    [Fact]
    public void InputDimensions_CanBeMultidimensional()
    {
        // Arrange & Act
        var operation = new TensorOperation
        {
            InputDimensions = new[] { 2, 3, 4, 5 }
        };

        // Assert
        Assert.Equal(4, operation.InputDimensions.Length);
        Assert.Equal(2, operation.InputDimensions[0]);
        Assert.Equal(5, operation.InputDimensions[3]);
    }
}
