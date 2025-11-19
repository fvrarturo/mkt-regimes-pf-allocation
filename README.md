# Macro Regimes, Cross-Asset Correlations, and Agentic AI for Dynamic Allocation

## Project Overview

This project develops a framework for identifying macroeconomic regimes and using them to explain and forecast stock-bond correlations. The approach combines traditional macroeconomic variables with text-derived sentiment scores from news articles, enabling regime-based dynamic asset allocation strategies.

## Goals

The project aims to:

1. **Identify Macro Regimes**: Classify economic conditions into distinct regimes (e.g., stagflation, overheating, recessionary stress) using both hard macro variables and text-based sentiment scores.

2. **Estimate Regime-Conditional Correlations**: Understand how stock-bond correlations behave within each macro regime, enabling better risk assessment and portfolio construction.

3. **Forecast Correlations**: Use regime probabilities to forecast future stock-bond correlations, supporting dynamic allocation decisions.

4. **Leverage Agentic AI**: Apply AI agents to process large news corpora, classify articles into macro categories, fact-check summaries, and generate sentiment scores aligned with economic theory.

## Methodology

The pipeline consists of four main stages:

1. **Feature Construction**: Build 8-dimensional feature vectors combining:
   - Macro variables: inflation, growth, monetary policy stance, financial stress
   - Text sentiment scores: AI-derived sentiment aligned with the same four macro categories

2. **Regime Classification**: Identify regimes using three approaches:
   - Manual (theory-driven): Economic logic-based regime definitions
   - ML-Found (unsupervised): Data-driven regime discovery via clustering
   - Guided (hybrid): Combination of economic intuition and data-driven methods

3. **Regime-Conditional Correlation Estimation**: Estimate how stock-bond correlations behave within each regime using weighted estimators based on soft regime probabilities.

4. **Evaluation**: Test regime-based forecasts using train/test splits that respect both time-dependence and regime coverage.

## Key Innovation

The integration of text-derived sentiment features with traditional macro variables allows for richer regime identification that captures both quantitative economic indicators and qualitative market narratives. This enables more nuanced understanding of how macro conditions affect cross-asset correlations and supports improved dynamic allocation strategies.

