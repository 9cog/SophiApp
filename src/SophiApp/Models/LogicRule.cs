// <copyright file="LogicRule.cs" company="Team Sophia">
// Copyright (c) Team Sophia. All rights reserved.
// </copyright>

namespace SophiApp.Models;

/// <summary>
/// Represents a symbolic logic rule in the Tensor Logic system.
/// </summary>
public class LogicRule
{
    /// <summary>
    /// Gets or sets the unique identifier for the logic rule.
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the logic rule.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the rule expression (e.g., "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)").
    /// </summary>
    public string Expression { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the predicate name.
    /// </summary>
    public string Predicate { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the antecedents (conditions) of the rule.
    /// </summary>
    public List<string> Antecedents { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the consequent (conclusion) of the rule.
    /// </summary>
    public string Consequent { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score (0.0 to 1.0) for probabilistic logic.
    /// </summary>
    public double Confidence { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets a value indicating whether this is a hard (exact) or soft (probabilistic) rule.
    /// </summary>
    public bool IsHardRule { get; set; } = true;

    /// <summary>
    /// Gets or sets the timestamp when the rule was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets metadata for the rule.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
}
