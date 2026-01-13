// <copyright file="TensorLogicServiceTests.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Tests.Services;

using SophiApp.Models;
using SophiApp.Services;
using Xunit;

/// <summary>
/// Unit tests for the TensorLogicService.
/// </summary>
public class TensorLogicServiceTests
{
    private readonly TensorLogicService service;

    public TensorLogicServiceTests()
    {
        this.service = new TensorLogicService();
    }

    #region CreateTensorOperation Tests

    [Fact]
    public void CreateTensorOperation_ShouldReturnValidOperation()
    {
        // Arrange
        var name = "test-tensor";
        var opType = "matmul";
        var dims = new[] { 2, 3 };
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

        // Act
        var result = this.service.CreateTensorOperation(name, opType, dims, data);

        // Assert
        Assert.NotNull(result);
        Assert.NotEqual(string.Empty, result.Id);
        Assert.Equal(name, result.Name);
        Assert.Equal(opType, result.OperationType);
        Assert.Equal(dims, result.InputDimensions);
        Assert.Equal(data, result.Data);
        Assert.NotNull(result.OutputDimensions);
    }

    [Fact]
    public void CreateTensorOperation_ShouldCalculateOutputDimensions()
    {
        // Arrange
        var dims = new[] { 3, 4 };
        var data = new double[12];

        // Act
        var result = this.service.CreateTensorOperation("test", "matmul", dims, data);

        // Assert
        Assert.NotNull(result.OutputDimensions);
        Assert.NotEmpty(result.OutputDimensions);
    }

    [Fact]
    public void CreateTensorOperation_ShouldSetCreatedAtTimestamp()
    {
        // Arrange
        var before = DateTime.UtcNow;

        // Act
        var result = this.service.CreateTensorOperation("test", "add", new[] { 2 }, new[] { 1.0, 2.0 });
        var after = DateTime.UtcNow;

        // Assert
        Assert.True(result.CreatedAt >= before);
        Assert.True(result.CreatedAt <= after);
    }

    #endregion

    #region CreateLogicRule Tests

    [Fact]
    public void CreateLogicRule_ShouldReturnValidRule()
    {
        // Arrange
        var name = "Grandparent Rule";
        var expression = "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)";
        var predicate = "grandparent";
        var antecedents = new List<string> { "parent(X,Y)", "parent(Y,Z)" };
        var consequent = "grandparent(X,Z)";

        // Act
        var result = this.service.CreateLogicRule(name, expression, predicate, antecedents, consequent);

        // Assert
        Assert.NotNull(result);
        Assert.NotEqual(string.Empty, result.Id);
        Assert.Equal(name, result.Name);
        Assert.Equal(expression, result.Expression);
        Assert.Equal(predicate, result.Predicate);
        Assert.Equal(antecedents, result.Antecedents);
        Assert.Equal(consequent, result.Consequent);
    }

    [Fact]
    public void CreateLogicRule_ShouldSetDefaultConfidence()
    {
        // Act
        var result = this.service.CreateLogicRule("test", "expr", "pred", new List<string>(), "cons");

        // Assert
        Assert.Equal(1.0, result.Confidence);
        Assert.True(result.IsHardRule);
    }

    #endregion

    #region CreateBridge Tests

    [Fact]
    public void CreateBridge_BooleanMode_ShouldDisableGradients()
    {
        // Act
        var result = this.service.CreateBridge("test", OperationMode.Boolean, 0.0);

        // Assert
        Assert.Equal(OperationMode.Boolean, result.Mode);
        Assert.False(result.EnableGradients);
        Assert.Equal(0.0, result.Temperature);
    }

    [Fact]
    public void CreateBridge_ContinuousMode_ShouldEnableGradients()
    {
        // Act
        var result = this.service.CreateBridge("test", OperationMode.Continuous, 1.5);

        // Assert
        Assert.Equal(OperationMode.Continuous, result.Mode);
        Assert.True(result.EnableGradients);
        Assert.Equal(1.5, result.Temperature);
    }

    [Fact]
    public void CreateBridge_ShouldSetDefaultEmbeddingDimension()
    {
        // Act
        var result = this.service.CreateBridge("test", OperationMode.Boolean, 0.0);

        // Assert
        Assert.Equal(64, result.EmbeddingDimension);
    }

    #endregion

    #region PerformEinsum Tests

    [Fact]
    public void PerformEinsum_MatrixMultiplication_ShouldCalculateCorrectly()
    {
        // Arrange
        var a = this.service.CreateTensorOperation("A", "matrix", new[] { 2, 2 }, new[] { 1.0, 2.0, 3.0, 4.0 });
        var b = this.service.CreateTensorOperation("B", "matrix", new[] { 2, 2 }, new[] { 5.0, 6.0, 7.0, 8.0 });

        // Act
        var result = this.service.PerformEinsum("ij,jk->ik", a, b);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("matmul", result.OperationType);
        Assert.Equal(4, result.Data.Length);
        // Matrix multiplication: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        Assert.Equal(19.0, result.Data[0]);
        Assert.Equal(22.0, result.Data[1]);
        Assert.Equal(43.0, result.Data[2]);
        Assert.Equal(50.0, result.Data[3]);
    }

    [Fact]
    public void PerformEinsum_NoOperands_ShouldThrowException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => this.service.PerformEinsum("ij,jk->ik"));
    }

    [Fact]
    public void PerformEinsum_InvalidEquation_ShouldThrowException()
    {
        // Arrange
        var tensor = this.service.CreateTensorOperation("test", "matrix", new[] { 2, 2 }, new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => this.service.PerformEinsum("invalid", tensor));
    }

    #endregion

    #region ApplyLogicRule Tests

    [Fact]
    public void ApplyLogicRule_WithValidFacts_ShouldDeriveFacts()
    {
        // Arrange
        var rule = this.service.CreateLogicRule(
            "Test Rule",
            "derived(X) :- condition(X)",
            "derived",
            new List<string> { "condition(X)" },
            "derived(X)");

        var facts = new List<TensorOperation>
        {
            this.service.CreateTensorOperation("derived_fact", "predicate", new[] { 10 }, new double[10])
        };

        // Act
        var result = this.service.ApplyLogicRule(rule, facts);

        // Assert
        Assert.NotNull(result);
        Assert.IsType<List<TensorOperation>>(result);
    }

    [Fact]
    public void ApplyLogicRule_WithEmptyFacts_ShouldReturnEmptyList()
    {
        // Arrange
        var rule = this.service.CreateLogicRule("test", "expr", "pred", new List<string> { "ant" }, "cons");
        var facts = new List<TensorOperation>();

        // Act
        var result = this.service.ApplyLogicRule(rule, facts);

        // Assert
        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void ApplyLogicRule_WithNullFacts_ShouldReturnEmptyList()
    {
        // Arrange
        var rule = this.service.CreateLogicRule("test", "expr", "pred", new List<string> { "ant" }, "cons");

        // Act
        var result = this.service.ApplyLogicRule(rule, null!);

        // Assert
        Assert.NotNull(result);
        Assert.Empty(result);
    }

    #endregion

    #region RuleToTensor Tests

    [Fact]
    public void RuleToTensor_ShouldConvertRuleToTensorOperations()
    {
        // Arrange
        var rule = this.service.CreateLogicRule(
            "Test Rule",
            "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)",
            "grandparent",
            new List<string> { "parent(X,Y)", "parent(Y,Z)" },
            "grandparent(X,Z)");

        // Act
        var result = this.service.RuleToTensor(rule);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result);
        // Should have tensors for each antecedent plus the consequent
        Assert.Equal(3, result.Count);
    }

    [Fact]
    public void RuleToTensor_ShouldCreateTensorsWithCorrectDimensions()
    {
        // Arrange
        var rule = this.service.CreateLogicRule(
            "Test",
            "test(X) :- cond(X)",
            "test",
            new List<string> { "cond(X)" },
            "test(X)");

        // Act
        var result = this.service.RuleToTensor(rule);

        // Assert
        foreach (var tensor in result)
        {
            Assert.NotNull(tensor.InputDimensions);
            Assert.NotNull(tensor.OutputDimensions);
            Assert.NotNull(tensor.Data);
            Assert.NotEmpty(tensor.Data);
        }
    }

    #endregion

    #region ForwardInference Tests

    [Fact]
    public void ForwardInference_BooleanMode_ShouldPerformExactInference()
    {
        // Arrange
        var bridge = this.service.CreateBridge("test", OperationMode.Boolean, 0.0);
        var query = this.service.CreateTensorOperation("query", "predicate", new[] { 4 }, new[] { 0.3, 0.7, 0.2, 0.9 });

        // Act
        var result = this.service.ForwardInference(bridge, query);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("boolean_inference", result.OperationType);
        // Values > 0.5 should become 1.0, others 0.0
        Assert.Equal(0.0, result.Data[0]);
        Assert.Equal(1.0, result.Data[1]);
        Assert.Equal(0.0, result.Data[2]);
        Assert.Equal(1.0, result.Data[3]);
    }

    [Fact]
    public void ForwardInference_ContinuousMode_ShouldPerformSoftInference()
    {
        // Arrange
        var bridge = this.service.CreateBridge("test", OperationMode.Continuous, 1.0);
        var query = this.service.CreateTensorOperation("query", "predicate", new[] { 3 }, new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = this.service.ForwardInference(bridge, query);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("continuous_inference", result.OperationType);
        // Result should be softmax-normalized probabilities
        var sum = result.Data.Sum();
        Assert.Equal(1.0, sum, precision: 5);
    }

    [Fact]
    public void ForwardInference_HybridMode_ShouldCombineBothApproaches()
    {
        // Arrange
        var bridge = this.service.CreateBridge("test", OperationMode.Hybrid, 0.5);
        var query = this.service.CreateTensorOperation("query", "predicate", new[] { 2 }, new[] { 0.3, 0.8 });

        // Act
        var result = this.service.ForwardInference(bridge, query);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("hybrid_inference", result.OperationType);
        Assert.Equal(2, result.Data.Length);
    }

    #endregion

    #region ComputeGradient Tests

    [Fact]
    public void ComputeGradient_DifferentiableOperation_ShouldCalculateGradient()
    {
        // Arrange
        var operation = this.service.CreateTensorOperation("test", "linear", new[] { 3 }, new[] { 1.0, 2.0, 3.0 });
        var target = new[] { 0.5, 1.5, 2.5 };

        // Act
        var result = this.service.ComputeGradient(operation, target);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("gradient", result.OperationType);
        Assert.Equal(0.5, result.Data[0]);
        Assert.Equal(0.5, result.Data[1]);
        Assert.Equal(0.5, result.Data[2]);
        Assert.False(result.IsDifferentiable);
    }

    [Fact]
    public void ComputeGradient_NonDifferentiableOperation_ShouldThrowException()
    {
        // Arrange
        var operation = this.service.CreateTensorOperation("test", "constant", new[] { 2 }, new[] { 1.0, 2.0 });
        operation.IsDifferentiable = false;
        var target = new[] { 1.0, 2.0 };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => this.service.ComputeGradient(operation, target));
    }

    #endregion

    #region ValidateTensorOperation Tests

    [Fact]
    public void ValidateTensorOperation_ValidOperation_ShouldReturnTrue()
    {
        // Arrange
        var operation = this.service.CreateTensorOperation("test", "add", new[] { 2, 2 }, new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var result = this.service.ValidateTensorOperation(operation);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ValidateTensorOperation_NullOperation_ShouldReturnFalse()
    {
        // Act
        var result = this.service.ValidateTensorOperation(null!);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateTensorOperation_EmptyName_ShouldReturnFalse()
    {
        // Arrange
        var operation = new TensorOperation
        {
            Name = "",
            OperationType = "test",
            InputDimensions = new[] { 2 },
            Data = new[] { 1.0, 2.0 }
        };

        // Act
        var result = this.service.ValidateTensorOperation(operation);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateTensorOperation_EmptyOperationType_ShouldReturnFalse()
    {
        // Arrange
        var operation = new TensorOperation
        {
            Name = "test",
            OperationType = "",
            InputDimensions = new[] { 2 },
            Data = new[] { 1.0, 2.0 }
        };

        // Act
        var result = this.service.ValidateTensorOperation(operation);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateTensorOperation_EmptyDimensions_ShouldReturnFalse()
    {
        // Arrange
        var operation = new TensorOperation
        {
            Name = "test",
            OperationType = "test",
            InputDimensions = Array.Empty<int>(),
            Data = new[] { 1.0, 2.0 }
        };

        // Act
        var result = this.service.ValidateTensorOperation(operation);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateTensorOperation_MismatchedDataSize_ShouldReturnFalse()
    {
        // Arrange
        var operation = new TensorOperation
        {
            Name = "test",
            OperationType = "test",
            InputDimensions = new[] { 2, 2 }, // Expects 4 elements
            Data = new[] { 1.0, 2.0 } // Only 2 elements
        };

        // Act
        var result = this.service.ValidateTensorOperation(operation);

        // Assert
        Assert.False(result);
    }

    #endregion

    #region ValidateLogicRule Tests

    [Fact]
    public void ValidateLogicRule_ValidRule_ShouldReturnTrue()
    {
        // Arrange
        var rule = this.service.CreateLogicRule(
            "Test Rule",
            "test(X) :- cond(X)",
            "test",
            new List<string> { "cond(X)" },
            "test(X)");

        // Act
        var result = this.service.ValidateLogicRule(rule);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ValidateLogicRule_NullRule_ShouldReturnFalse()
    {
        // Act
        var result = this.service.ValidateLogicRule(null!);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateLogicRule_EmptyName_ShouldReturnFalse()
    {
        // Arrange
        var rule = new LogicRule
        {
            Name = "",
            Expression = "expr",
            Predicate = "pred",
            Antecedents = new List<string> { "ant" },
            Consequent = "cons"
        };

        // Act
        var result = this.service.ValidateLogicRule(rule);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateLogicRule_EmptyExpression_ShouldReturnFalse()
    {
        // Arrange
        var rule = new LogicRule
        {
            Name = "test",
            Expression = "",
            Predicate = "pred",
            Antecedents = new List<string> { "ant" },
            Consequent = "cons"
        };

        // Act
        var result = this.service.ValidateLogicRule(rule);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateLogicRule_EmptyAntecedents_ShouldReturnFalse()
    {
        // Arrange
        var rule = new LogicRule
        {
            Name = "test",
            Expression = "expr",
            Predicate = "pred",
            Antecedents = new List<string>(),
            Consequent = "cons"
        };

        // Act
        var result = this.service.ValidateLogicRule(rule);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateLogicRule_InvalidConfidence_ShouldReturnFalse()
    {
        // Arrange
        var rule = new LogicRule
        {
            Name = "test",
            Expression = "expr",
            Predicate = "pred",
            Antecedents = new List<string> { "ant" },
            Consequent = "cons",
            Confidence = 1.5 // Invalid: > 1.0
        };

        // Act
        var result = this.service.ValidateLogicRule(rule);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateLogicRule_NegativeConfidence_ShouldReturnFalse()
    {
        // Arrange
        var rule = new LogicRule
        {
            Name = "test",
            Expression = "expr",
            Predicate = "pred",
            Antecedents = new List<string> { "ant" },
            Consequent = "cons",
            Confidence = -0.5 // Invalid: < 0.0
        };

        // Act
        var result = this.service.ValidateLogicRule(rule);

        // Assert
        Assert.False(result);
    }

    #endregion
}
