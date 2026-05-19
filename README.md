#  Data Breach Trend Analysis (2004–2023)

**University at Albany | Oct 2025 – Dec 2025**

## What is this project?

This project analyzed 389 real data breach records from around the world covering 20 years between 2004 and 2023. The goal was to find patterns — which industries get attacked the most, which attack methods cause the most damage, and whether breach severity is getting worse over time. I used Python to clean the data, create visualizations, and build two statistical regression models.

## What did I find?

The Web, Government, Financial Services, and Technology sectors had the highest number of records exposed. Hacking and poor security practices together caused over 80 percent of all breaches globally. The regression model confirmed that breach severity is increasing by approximately 4.6 million additional records exposed per year. The p-value of 0.039 confirmed this trend is statistically significant.

## Models I built

I built a Simple Linear Regression using Year as the predictor variable, and a Multiple Linear Regression adding Organization Type as a second predictor. The simple model gave an R² of 0.011 and the multiple model gave R² of 0.0166, confirming that both time and industry type influence breach severity.

## Tools I used

Python · Pandas · Matplotlib · Seaborn · Scikit-learn · Statsmodels

## Files in this project

`data_breach_analysis.py` — the complete Python analysis script with all models, EDA, and charts

`figures_and_analysis.pdf` — all charts and regression output figures with written analysis

## Skills this shows

Data analysis · Exploratory data analysis (EDA) · Regression modeling · Python · Data visualization · Cybersecurity risk assessment · Threat intelligence
