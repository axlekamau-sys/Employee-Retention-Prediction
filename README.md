# Employee Retention Prediction Project

## Project Overview

This project develops a machine learning solution to predict employee turnover, enabling HR teams to identify at-risk employees and implement proactive retention strategies. The system analyzes various employee attributes including satisfaction levels, project involvement, work hours, and compensation to determine the likelihood of an employee leaving the organization.

### Key Features
-  **99.2% accurate** XGBoost-based prediction model
-  Comprehensive Exploratory Data Analysis
-  Feature importance analysis for model interpretability
-  Risk segmentation framework (Low/Medium/High risk)
-  Actionable business recommendations

---

##  Business Objectives

### Main Objective
To develop a predictive model that accurately identifies employees at risk of leaving and provides actionable insights for HR decision-making.

### Specific Objectives
1. Identify key factors associated with employee turnover
2. Build accurate prediction models using various algorithms
3. Create risk profiles for different employee segments
4. Develop actionable recommendations for retention strategies

### Success Criteria
- **Accuracy**: >90% on test data
- **Interpretability**: Clear identification of key drivers
- **Business Impact**: Actionable insights for HR strategies

---

##  Dataset Overview

 Attribute & Description

**Source**  HR database (anonymized) 
**Records**  14,999 employees 
**Features**  9 predictors + 1 target 
**Target**  `left` (0 = stayed, 1 = left) 

### Feature Description

 Feature & Type & Description 

satisfaction_level  Float  Employee satisfaction score (0-1) 
 last_evaluation  Float  Last performance score (0-1) 
 number_project  Integer  Number of projects assigned 
| average_monthly_hours  Integer  Average hours worked/month 
| time_spend_company  Integer  Years at company 
| Work_accident  Binary  Had work accident (0/1) 
 promotion_last_5years  Binary  Got promoted (0/1) 
 salary  Categorical  low/medium/high 
  left  Binary  **TARGET** 

---

## 🔍 Key Insights from EDA

### 1. Target Distribution
- 76% stayed, 24% left (class imbalance)

### 2. Salary Impact
- Low salary: 60% attrition rate
- Medium salary: 45% attrition rate
- High salary: 15% attrition rate
- **Low salary employees are 4x more likely to leave**

### 3. Promotion Impact
- Promoted: Only 5% left
- Not promoted: 25% left
- **No promotion = 5x higher attrition risk**

### 4. Satisfaction Level
- Stayers: Median satisfaction ~0.65
- Leavers: Median satisfaction ~0.45
- **Satisfaction is the strongest predictor**

### 5. Workload Analysis
- 3-4 projects optimal for retention
- 6+ projects shows burnout risk
- 250+ hours/month leads to attrition

### 6. Critical Tenure Period
- 3-5 year mark shows highest attrition

---

## 🤖 Model Performance Comparison

 Model & Accuracy & Precision & Recall & F1-Score 

 Logistic Regression | 78.4% | 0.75 | 0.71 | 0.73 |
 Random Forest | 92.3% | 0.92 | 0.90 | 0.91 |

 **XGBoost** | **99.2%** | **0.99** | **0.98** | **0.98** |

### XGBoost Confusion Matrix

- **True Negatives**: 2,292
- **True Positives**: 685
- **False Positives**: 7
- **False Negatives**: 16

---

## 📈 Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | satisfaction_level | 0.32 |
| 2 | number_project | 0.24 |
| 3 | time_spend_company | 0.18 |
| 4 | average_monthly_hours | 0.14 |
| 5 | last_evaluation | 0.08 |

---

## 🚦 Risk Segmentation Framework

### 🟢 LOW RISK (0-30% probability)
- **Profile**: High satisfaction, 3-4 projects, promoted, medium/high salary
- **Action**: Regular engagement, development plans

### 🟡 MEDIUM RISK (30-70% probability)
- **Profile**: Medium satisfaction, 4-5 projects, 2-4 years tenure, no promotion
- **Action**: Stay interviews, career discussions, workload check

### 🔴 HIGH RISK (70%+ probability)
- **Profile**: Low satisfaction, 6+ projects, 3-5 years tenure, low salary, no promotion
- **Action**: IMMEDIATE intervention, manager discussion, retention bonus

---

## 💡 Business Recommendations

### Short-Term Actions
1. **Targeted Retention Program**
   - Identify employees with satisfaction < 0.5
   - Conduct stay interviews
   - Address specific concerns

2. **Workload Management**
   - Monitor employees with 6+ projects
   - Redistribute work if needed
   - Enforce work-life balance

### Long-Term Strategies
3. **Career Development Path**
   - Clear promotion criteria
   - Regular career discussions
   - Skill development programs

4. **Compensation Review**
   - Competitive salary benchmarking
   - Performance-based incentives
   - Retention bonuses for key talent

### Implementation Roadmap
- **Month 1**: Deploy model & identify at-risk
- **Month 2**: Begin targeted interventions
- **Month 3**: Track progress & adjust
- **Month 6**: Measure impact & refine

---

## 🛠️ Technical Implementation

### Requirements