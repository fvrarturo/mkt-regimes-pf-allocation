Read the entire directory of this project. 
The goal is to do an exploratory analysis of macro variables' impact on equity risk premium, forecast macro variables, and test trading strategies in the context of asset allocation
The project is divided into 3 parts:
1. Analysis of which macro variables have the most predictability on Equity risk Premium
    - To do so we used the lens of defining regimes, extremeness to do conditional subregressions
    - We're looking for the regime/extremeness definition that shows the clearest patterns in terms of significance of linear regression coef for macro variables
2. Then, we forecast industrial production and inflation as two variables very useful in this pattern recognition
    - In here we test a bunch of models and see which one performs best
    - As one of the tasks asks to see the benefits of Agentic AI, we test them by adding sentiment scores derived from news data using an agentic AI framework (an agent that reads news, etc.)
    - WE see which model performs best
3. Finally, we test trading strategies in the context of asset allocation:
    - As we couldn't predict all macro variables within a reasonable timeline, we assume that we can forecast macro variables for different accuracy thresholds (40%, 60%, 80%, 100%)
    - Based on this assumption (i.e. data), we use different models (HMM regimeness, 2x2 regimeness, extremeness, full sample etc.) to see which one yields the best forecasts of ERP
    - In terms of pure trading strategy, we implement simple ones (we have ERP now and a forecast of erp in the future so we need to reallocate weights between sp500 and tbills)
        - we can use binary (100% sp500 or 100% tbills) and weighted
    - We also a benchmark strategy of fixed 60/40

Your goal is to analyze the full directory, organize better files (removing redundancies) to follow this pipeline, improve models in forecasting, regimeness, extremeness, and build the trading rule

This is a very extensive task in terms of thinking.
