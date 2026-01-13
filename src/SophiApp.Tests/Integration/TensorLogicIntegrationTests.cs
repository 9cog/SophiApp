// <copyright file="TensorLogicIntegrationTests.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Tests.Integration;

using SophiApp.Models;
using SophiApp.Services;

/// <summary>
/// Integration tests demonstrating real-world Tensor Logic scenarios.
/// </summary>
public class TensorLogicIntegrationTests
{
    private readonly TensorLogicService service;

    public TensorLogicIntegrationTests()
    {
        this.service = new TensorLogicService();
    }

    [Fact]
    public void Scenario_FamilyTreeReasoning_ShouldDeriveGrandparentRelations()
    {
        // Arrange: Create parent relationships as tensors
        // Alice -> Bob, Bob -> Charlie
        var parentAliceBob = this.service.CreateTensorOperation(
            "parent_alice_bob",
            "predicate",
            new[] { 2, 2 },
            new[] { 0.0, 1.0, 0.0, 0.0 }); // Alice is parent of Bob

        var parentBobCharlie = this.service.CreateTensorOperation(
            "parent_bob_charlie",
            "predicate",
            new[] { 2, 2 },
            new[] { 0.0, 0.0, 0.0, 1.0 }); // Bob is parent of Charlie

        // Create grandparent rule
        var grandparentRule = this.service.CreateLogicRule(
            "Grandparent Rule",
            "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)",
            "grandparent",
            new List<string> { "parent(X,Y)", "parent(Y,Z)" },
            "grandparent(X,Z)");

        // Act: Apply rule through tensor multiplication (composing relations)
        var grandparent = this.service.PerformEinsum("ij,jk->ik", parentAliceBob, parentBobCharlie);

        // Assert: Alice should be grandparent of Charlie
        Assert.NotNull(grandparent);
        Assert.Equal("matmul", grandparent.OperationType);
        Assert.True(this.service.ValidateTensorOperation(grandparent));
    }

    [Fact]
    public void Scenario_TemperatureControl_ShouldAdjustInferenceStrictness()
    {
        // Arrange: Ambiguous query with uncertain values
        var ambiguousQuery = this.service.CreateTensorOperation(
            "ambiguous_query",
            "predicate",
            new[] { 4 },
            new[] { 0.45, 0.55, 0.48, 0.52 });

        var strictBridge = this.service.CreateBridge("strict", OperationMode.Boolean, 0.0);
        var relaxedBridge = this.service.CreateBridge("relaxed", OperationMode.Continuous, 2.0);

        // Act
        var strictResult = this.service.ForwardInference(strictBridge, ambiguousQuery);
        var relaxedResult = this.service.ForwardInference(relaxedBridge, ambiguousQuery);

        // Assert: Strict mode makes hard decisions, relaxed mode preserves uncertainty
        Assert.NotNull(strictResult);
        Assert.NotNull(relaxedResult);

        // Strict mode should have binary values (0 or 1)
        Assert.All(strictResult.Data, value => Assert.True(value == 0.0 || value == 1.0));

        // Relaxed mode should have probabilistic values
        var sum = relaxedResult.Data.Sum();
        Assert.Equal(1.0, sum, precision: 5);
    }

    [Fact]
    public void Scenario_LearningFromData_ShouldComputeGradientsCorrectly()
    {
        // Arrange: System making predictions that need correction
        var prediction = this.service.CreateTensorOperation(
            "prediction",
            "linear",
            new[] { 5 },
            new[] { 0.9, 0.1, 0.8, 0.3, 0.7 });

        var groundTruth = new[] { 1.0, 0.0, 1.0, 0.0, 1.0 };

        // Act: Compute gradient to learn from mistake
        var gradient = this.service.ComputeGradient(prediction, groundTruth);

        // Assert: Gradient points in direction to improve predictions
        Assert.NotNull(gradient);
        Assert.Equal(5, gradient.Data.Length);

        // Check gradient magnitude is reasonable
        var avgGradient = gradient.Data.Average();
        Assert.True(Math.Abs(avgGradient) < 1.0);
    }

    [Fact]
    public void Scenario_ComplexTensorOperations_ShouldHandleMatrixChains()
    {
        // Arrange: Create a series of matrices to multiply
        var m1 = this.service.CreateTensorOperation(
            "M1",
            "matrix",
            new[] { 2, 3 },
            new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

        var m2 = this.service.CreateTensorOperation(
            "M2",
            "matrix",
            new[] { 3, 2 },
            new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

        // Act: (2x3) * (3x2) = (2x2)
        var result = this.service.PerformEinsum("ij,jk->ik", m1, m2);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(new[] { 2, 2 }, result.InputDimensions);
        Assert.Equal(4, result.Data.Length);

        // Verify matrix multiplication is correct
        // [[1,2,3],[4,5,6]] * [[1,2],[3,4],[5,6]]
        // = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        // = [[22, 28], [49, 64]]
        Assert.Equal(22.0, result.Data[0]);
        Assert.Equal(28.0, result.Data[1]);
        Assert.Equal(49.0, result.Data[2]);
        Assert.Equal(64.0, result.Data[3]);
    }

    [Fact]
    public void Scenario_ValidationPipeline_ShouldCatchInvalidOperations()
    {
        // Arrange: Create various valid and invalid operations
        var validOp = this.service.CreateTensorOperation("valid", "add", new[] { 2, 2 }, new[] { 1.0, 2.0, 3.0, 4.0 });

        var invalidOp1 = new TensorOperation
        {
            Name = "",  // Invalid: empty name
            OperationType = "test",
            InputDimensions = new[] { 2 },
            Data = new[] { 1.0, 2.0 }
        };

        var invalidOp2 = new TensorOperation
        {
            Name = "test",
            OperationType = "test",
            InputDimensions = new[] { 2, 2 },  // Says 2x2 = 4 elements
            Data = new[] { 1.0, 2.0 }  // Only 2 elements!
        };

        // Act
        var validResult = this.service.ValidateTensorOperation(validOp);
        var invalid1Result = this.service.ValidateTensorOperation(invalidOp1);
        var invalid2Result = this.service.ValidateTensorOperation(invalidOp2);

        // Assert
        Assert.True(validResult);
        Assert.False(invalid1Result);
        Assert.False(invalid2Result);
    }
}
