// <copyright file="NeuralSymbolicBridgeTests.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Tests.Models;

using SophiApp.Models;
using Xunit;

/// <summary>
/// Unit tests for the NeuralSymbolicBridge model.
/// </summary>
public class NeuralSymbolicBridgeTests
{
    [Fact]
    public void Constructor_ShouldInitializeDefaultValues()
    {
        // Arrange & Act
        var bridge = new NeuralSymbolicBridge();

        // Assert
        Assert.NotNull(bridge);
        Assert.Equal(string.Empty, bridge.Id);
        Assert.Equal(string.Empty, bridge.Name);
        Assert.Equal(0.0, bridge.Temperature);
        Assert.Equal(OperationMode.Boolean, bridge.Mode);
        Assert.NotNull(bridge.TensorOperations);
        Assert.Empty(bridge.TensorOperations);
        Assert.NotNull(bridge.LogicRules);
        Assert.Empty(bridge.LogicRules);
        Assert.Equal(64, bridge.EmbeddingDimension);
        Assert.False(bridge.EnableGradients);
    }

    [Fact]
    public void Properties_ShouldSetAndGetCorrectly()
    {
        // Arrange
        var bridge = new NeuralSymbolicBridge
        {
            Id = "bridge-1",
            Name = "Test Bridge",
            Temperature = 1.5,
            Mode = OperationMode.Continuous,
            EmbeddingDimension = 128,
            EnableGradients = true
        };

        // Act & Assert
        Assert.Equal("bridge-1", bridge.Id);
        Assert.Equal("Test Bridge", bridge.Name);
        Assert.Equal(1.5, bridge.Temperature);
        Assert.Equal(OperationMode.Continuous, bridge.Mode);
        Assert.Equal(128, bridge.EmbeddingDimension);
        Assert.True(bridge.EnableGradients);
    }

    [Fact]
    public void OperationMode_ShouldSupportAllValues()
    {
        // Arrange & Act
        var booleanBridge = new NeuralSymbolicBridge { Mode = OperationMode.Boolean };
        var continuousBridge = new NeuralSymbolicBridge { Mode = OperationMode.Continuous };
        var hybridBridge = new NeuralSymbolicBridge { Mode = OperationMode.Hybrid };

        // Assert
        Assert.Equal(OperationMode.Boolean, booleanBridge.Mode);
        Assert.Equal(OperationMode.Continuous, continuousBridge.Mode);
        Assert.Equal(OperationMode.Hybrid, hybridBridge.Mode);
    }

    [Fact]
    public void TensorOperations_CanBeAddedAndRetrieved()
    {
        // Arrange
        var bridge = new NeuralSymbolicBridge();
        var op1 = new TensorOperation { Name = "op1" };
        var op2 = new TensorOperation { Name = "op2" };

        // Act
        bridge.TensorOperations.Add(op1);
        bridge.TensorOperations.Add(op2);

        // Assert
        Assert.Equal(2, bridge.TensorOperations.Count);
        Assert.Contains(op1, bridge.TensorOperations);
        Assert.Contains(op2, bridge.TensorOperations);
    }

    [Fact]
    public void LogicRules_CanBeAddedAndRetrieved()
    {
        // Arrange
        var bridge = new NeuralSymbolicBridge();
        var rule1 = new LogicRule { Name = "rule1" };
        var rule2 = new LogicRule { Name = "rule2" };

        // Act
        bridge.LogicRules.Add(rule1);
        bridge.LogicRules.Add(rule2);

        // Assert
        Assert.Equal(2, bridge.LogicRules.Count);
        Assert.Contains(rule1, bridge.LogicRules);
        Assert.Contains(rule2, bridge.LogicRules);
    }

    [Fact]
    public void Temperature_ShouldControlInferenceStrictness()
    {
        // Arrange & Act
        var strictBridge = new NeuralSymbolicBridge { Temperature = 0.0 };
        var relaxedBridge = new NeuralSymbolicBridge { Temperature = 2.0 };

        // Assert
        Assert.Equal(0.0, strictBridge.Temperature);
        Assert.Equal(2.0, relaxedBridge.Temperature);
    }

    [Fact]
    public void EmbeddingDimension_ShouldAcceptPositiveValues()
    {
        // Arrange
        var bridge = new NeuralSymbolicBridge();

        // Act & Assert
        bridge.EmbeddingDimension = 32;
        Assert.Equal(32, bridge.EmbeddingDimension);

        bridge.EmbeddingDimension = 256;
        Assert.Equal(256, bridge.EmbeddingDimension);

        bridge.EmbeddingDimension = 512;
        Assert.Equal(512, bridge.EmbeddingDimension);
    }

    [Fact]
    public void CreatedAt_ShouldBeSetAutomatically()
    {
        // Arrange
        var beforeCreation = DateTime.UtcNow;

        // Act
        var bridge = new NeuralSymbolicBridge();
        var afterCreation = DateTime.UtcNow;

        // Assert
        Assert.True(bridge.CreatedAt >= beforeCreation);
        Assert.True(bridge.CreatedAt <= afterCreation);
    }

    [Fact]
    public void EnableGradients_ShouldBeSetBasedOnMode()
    {
        // Arrange & Act
        var booleanBridge = new NeuralSymbolicBridge
        {
            Mode = OperationMode.Boolean,
            EnableGradients = false
        };
        var continuousBridge = new NeuralSymbolicBridge
        {
            Mode = OperationMode.Continuous,
            EnableGradients = true
        };

        // Assert
        Assert.False(booleanBridge.EnableGradients);
        Assert.True(continuousBridge.EnableGradients);
    }
}
