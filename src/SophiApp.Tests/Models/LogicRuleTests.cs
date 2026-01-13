// <copyright file="LogicRuleTests.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Tests.Models;

using SophiApp.Models;
using Xunit;

/// <summary>
/// Unit tests for the LogicRule model.
/// </summary>
public class LogicRuleTests
{
    [Fact]
    public void Constructor_ShouldInitializeDefaultValues()
    {
        // Arrange & Act
        var rule = new LogicRule();

        // Assert
        Assert.NotNull(rule);
        Assert.Equal(string.Empty, rule.Id);
        Assert.Equal(string.Empty, rule.Name);
        Assert.Equal(string.Empty, rule.Expression);
        Assert.Equal(string.Empty, rule.Predicate);
        Assert.NotNull(rule.Antecedents);
        Assert.Empty(rule.Antecedents);
        Assert.Equal(string.Empty, rule.Consequent);
        Assert.Equal(1.0, rule.Confidence);
        Assert.True(rule.IsHardRule);
        Assert.NotNull(rule.Metadata);
        Assert.Empty(rule.Metadata);
    }

    [Fact]
    public void Properties_ShouldSetAndGetCorrectly()
    {
        // Arrange
        var rule = new LogicRule
        {
            Id = "rule-1",
            Name = "Grandparent Rule",
            Expression = "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)",
            Predicate = "grandparent",
            Antecedents = new List<string> { "parent(X,Y)", "parent(Y,Z)" },
            Consequent = "grandparent(X,Z)",
            Confidence = 0.95,
            IsHardRule = false
        };

        // Act & Assert
        Assert.Equal("rule-1", rule.Id);
        Assert.Equal("Grandparent Rule", rule.Name);
        Assert.Equal("grandparent(X,Z) :- parent(X,Y), parent(Y,Z)", rule.Expression);
        Assert.Equal("grandparent", rule.Predicate);
        Assert.Equal(2, rule.Antecedents.Count);
        Assert.Equal("grandparent(X,Z)", rule.Consequent);
        Assert.Equal(0.95, rule.Confidence);
        Assert.False(rule.IsHardRule);
    }

    [Fact]
    public void Antecedents_CanBeModified()
    {
        // Arrange
        var rule = new LogicRule();

        // Act
        rule.Antecedents.Add("condition1");
        rule.Antecedents.Add("condition2");
        rule.Antecedents.Add("condition3");

        // Assert
        Assert.Equal(3, rule.Antecedents.Count);
        Assert.Contains("condition1", rule.Antecedents);
        Assert.Contains("condition2", rule.Antecedents);
        Assert.Contains("condition3", rule.Antecedents);
    }

    [Fact]
    public void Confidence_ShouldAcceptValidRange()
    {
        // Arrange
        var rule = new LogicRule();

        // Act & Assert
        rule.Confidence = 0.0;
        Assert.Equal(0.0, rule.Confidence);

        rule.Confidence = 0.5;
        Assert.Equal(0.5, rule.Confidence);

        rule.Confidence = 1.0;
        Assert.Equal(1.0, rule.Confidence);
    }

    [Fact]
    public void Metadata_ShouldStoreAdditionalInformation()
    {
        // Arrange
        var rule = new LogicRule();

        // Act
        rule.Metadata["author"] = "John Doe";
        rule.Metadata["version"] = 2;
        rule.Metadata["tags"] = new[] { "family", "relations" };

        // Assert
        Assert.Equal(3, rule.Metadata.Count);
        Assert.Equal("John Doe", rule.Metadata["author"]);
        Assert.Equal(2, rule.Metadata["version"]);
    }

    [Fact]
    public void CreatedAt_ShouldBeSetAutomatically()
    {
        // Arrange
        var beforeCreation = DateTime.UtcNow;

        // Act
        var rule = new LogicRule();
        var afterCreation = DateTime.UtcNow;

        // Assert
        Assert.True(rule.CreatedAt >= beforeCreation);
        Assert.True(rule.CreatedAt <= afterCreation);
    }

    [Fact]
    public void IsHardRule_ShouldDistinguishRuleTypes()
    {
        // Arrange
        var hardRule = new LogicRule { IsHardRule = true, Confidence = 1.0 };
        var softRule = new LogicRule { IsHardRule = false, Confidence = 0.8 };

        // Act & Assert
        Assert.True(hardRule.IsHardRule);
        Assert.Equal(1.0, hardRule.Confidence);
        Assert.False(softRule.IsHardRule);
        Assert.Equal(0.8, softRule.Confidence);
    }
}
