# Collections NBA Feature Engineering Guide

> **Comprehensive guide to features for Next Best Action (NBA) in collections recovery**

---

## 📋 Table of Contents

1. [Feature Categories Overview](#feature-categories-overview)
2. [Detailed Feature Taxonomy](#detailed-feature-taxonomy)
3. [Feature Engineering Best Practices](#feature-engineering-best-practices)
4. [Critical Features for NBA](#critical-features-for-nba)
5. [Schema Requirements](#schema-requirements)

---

## 🎯 Feature Categories Overview

For effective Collections NBA, we engineer **150+ features** across 9 major categories:

### 1. **Account & Delinquency Features** (15-20 features)
Track account status, delinquency depth, and portfolio characteristics

### 2. **Behavioral Features** (40-50 features)
Velocity, momentum, volatility, and stability of financial behaviors

### 3. **Payment History Features** (25-30 features)
Historical payment patterns, promises, and fulfillment

### 4. **Contact & Response Features** (20-25 features)
Communication history, channel preferences, contact outcomes

### 5. **Liquidity & Financial Health** (15-20 features)
OTB availability, utilization, income stability, debt burden

### 6. **Temporal & Seasonality** (10-15 features)
Day of week, month, payroll cycles, seasonal patterns

### 7. **Treatment Response** (15-20 features)
Historical response to different collection strategies

### 8. **External & Macro Features** (5-10 features)
Economic indicators, regional factors, industry trends

### 9. **Derived Risk Features** (10-15 features)
Composite scores, propensity predictions, expected values

**Total: 150+ engineered features**

---

## 📊 Detailed Feature Taxonomy

### Category 1: Account & Delinquency Features

**Core Account Metrics:**
- `principal_balance` - Current outstanding principal
- `total_balance` - Total amount due (principal + interest + fees)
- `current_dpd` - Days Past Due (current)
- `max_dpd_ever` - Highest DPD in account history
- `cycle_dpd` - DPD at each billing cycle
- `delinquency_bucket` - 0-30, 31-60, 61-90, 91-180, 180+
- `account_age_days` - Days since account opening
- `account_age_months` - Months since account opening
- `product_type` - Credit card, personal loan, auto loan, etc.
- `credit_limit` - Original credit limit (for revolving products)
- `original_loan_amount` - Original disbursement amount
- `remaining_tenure_months` - Months remaining to maturity
- `interest_rate` - Current interest rate
- `total_fees_charged` - Cumulative fees
- `write_off_risk_score` - Internal write-off probability

**Delinquency Progression:**
- `dpd_trend_7d` - DPD change in last 7 days
- `dpd_trend_30d` - DPD change in last 30 days
- `dpd_acceleration` - Rate of DPD increase
- `time_in_current_bucket` - Days in current delinquency bucket
- `bucket_transitions_count` - Number of bucket movements
- `rolled_back_count` - Number of times cured and re-defaulted

---

### Category 2: Behavioral Features

**Velocity Features (Rate of Change):**
- `payment_velocity_7d` - Payment frequency last 7 days
- `payment_velocity_30d` - Payment frequency last 30 days
- `payment_velocity_90d` - Payment frequency last 90 days
- `transaction_velocity_7d` - Transaction count last 7 days
- `transaction_velocity_30d` - Transaction count last 30 days
- `spend_velocity_30d` - Spending rate last 30 days
- `balance_velocity_30d` - Balance change rate
- `dpd_velocity` - DPD change velocity

**Momentum Features (Acceleration):**
- `payment_momentum_30d` - Acceleration in payment frequency
- `spend_momentum_30d` - Acceleration in spending
- `balance_momentum` - Acceleration in balance changes
- `dpd_momentum` - Acceleration in delinquency

**Volatility Features (Instability):**
- `payment_volatility_30d` - Std dev of payment amounts (30d)
- `payment_volatility_90d` - Std dev of payment amounts (90d)
- `payment_cv_30d` - Coefficient of variation in payments
- `transaction_volatility` - Std dev in transaction counts
- `income_volatility` - Std dev in deposit patterns
- `spend_volatility` - Std dev in spending

**Stability Features (Consistency):**
- `payment_regularity_score` - How regular are payments
- `payment_day_consistency` - Do they pay on same day each month
- `payment_amount_consistency` - Payment amount consistency
- `behavioral_stability_index` - Overall stability metric (0-1)
- `trend_strength` - Strength of behavioral trend

**Elasticity Features (Sensitivity):**
- `payment_sensitivity_to_contact` - Response to contact attempts
- `payment_sensitivity_to_reminder` - Response to reminders
- `payment_sensitivity_to_dpd` - How DPD affects behavior
- `spend_sensitivity_to_balance` - Spending response to balance

---

### Category 3: Payment History Features

**Payment Patterns:**
- `total_payments_count` - Total payments made (lifetime)
- `payments_count_30d` - Payments in last 30 days
- `payments_count_90d` - Payments in last 90 days
- `payments_count_180d` - Payments in last 180 days
- `avg_payment_amount_30d` - Average payment (30d)
- `avg_payment_amount_90d` - Average payment (90d)
- `median_payment_amount` - Median payment amount
- `max_payment_amount` - Largest single payment
- `min_payment_amount` - Smallest payment
- `total_paid_amount` - Cumulative amount paid

**Payment Timing:**
- `days_since_last_payment` - Recency of last payment
- `avg_days_between_payments` - Average payment frequency
- `payment_day_of_month` - Typical payment day
- `payment_day_consistency_score` - Day consistency (0-1)
- `early_payment_ratio` - % of payments before due date
- `late_payment_ratio` - % of payments after due date

**Promise & Fulfillment:**
- `promise_to_pay_count` - Total promises made
- `promise_kept_count` - Promises fulfilled
- `promise_broken_count` - Promises broken
- `promise_kept_ratio` - Promise keeping rate
- `avg_promise_amount` - Average promised amount
- `avg_promise_fulfillment_days` - Days to fulfill promise
- `broken_promise_streak` - Consecutive broken promises
- `days_since_last_promise` - Recency of last promise

**Partial Payments:**
- `partial_payment_count` - Number of partial payments
- `partial_payment_ratio` - % of payments that are partial
- `avg_partial_payment_pct` - Average % paid when partial
- `min_payment_only_count` - Times only min payment made

---

### Category 4: Contact & Response Features

**Contact History:**
- `total_contact_attempts` - Total contact attempts (lifetime)
- `contact_attempts_7d` - Contact attempts last 7 days
- `contact_attempts_30d` - Contact attempts last 30 days
- `successful_contacts_count` - Successful connections
- `contact_success_rate` - % of successful contacts
- `days_since_last_contact` - Recency of last contact
- `days_since_successful_contact` - Recency of successful contact

**Channel Preferences:**
- `phone_contact_count` - Phone calls count
- `sms_contact_count` - SMS count
- `email_contact_count` - Email count
- `whatsapp_contact_count` - WhatsApp messages count
- `preferred_channel` - Most responsive channel
- `channel_response_rate_phone` - Phone response rate
- `channel_response_rate_sms` - SMS response rate
- `channel_response_rate_email` - Email response rate

**Contact Outcomes:**
- `right_party_contact_count` - Contacts with right person
- `wrong_number_count` - Wrong number attempts
- `no_answer_count` - No answer count
- `busy_declined_count` - Busy/declined calls
- `voicemail_count` - Voicemails left
- `callback_requested_count` - Callbacks requested
- `dispute_raised_count` - Disputes raised

**Response Quality:**
- `avg_call_duration` - Average call duration (seconds)
- `engagement_score` - Overall engagement quality (0-1)
- `hostile_interaction_count` - Hostile/aggressive interactions
- `cooperative_interaction_count` - Cooperative interactions
- `financial_hardship_flagged` - Hardship indicated

---

### Category 5: Liquidity & Financial Health

**OTB (On-The-Book) Features:**
- `otb_available` - Available credit limit
- `otb_utilization_ratio` - Utilization % of available credit
- `otb_utilization_trend_30d` - Change in utilization (30d)
- `otb_velocity` - Rate of change in OTB
- `otb_stability` - Consistency of OTB usage
- `otb_momentum` - Acceleration in OTB usage

**Repayment Capacity:**
- `estimated_monthly_income` - Estimated income
- `deposit_amount_30d` - Total deposits last 30 days
- `deposit_frequency_30d` - Deposit frequency
- `deposit_regularity_score` - Income regularity (0-1)
- `income_volatility` - Std dev in deposits
- `debt_to_income_ratio` - Total debt / income
- `free_cash_flow_estimate` - Income - expenses estimate

**Utilization Patterns:**
- `current_utilization` - Current balance / limit
- `max_utilization_ever` - Peak utilization
- `avg_utilization_30d` - Average utilization (30d)
- `avg_utilization_90d` - Average utilization (90d)
- `utilization_trend` - Direction of utilization change

**Installment Lock (for EMI products):**
- `emi_amount` - Monthly EMI amount
- `emi_to_income_ratio` - EMI burden as % of income
- `emi_payment_regularity` - Regularity of EMI payments
- `emi_payment_count_on_time` - On-time EMI payments
- `installment_volatility` - Variability in EMI payments

---

### Category 6: Temporal & Seasonality Features

**Time-Based Features:**
- `day_of_week` - Day of week (0-6)
- `day_of_month` - Day of month (1-31)
- `week_of_month` - Week of month (1-4)
- `month` - Month (1-12)
- `quarter` - Quarter (1-4)
- `is_weekend` - Weekend flag
- `is_month_end` - Month-end flag (last 3 days)
- `is_month_start` - Month-start flag (first 3 days)

**Payroll Cycles:**
- `days_since_salary_date` - Days from typical payday
- `days_to_next_salary` - Days to next payday
- `is_payday_week` - Week containing typical payday
- `payday_proximity_score` - Proximity to payday (0-1)

**Seasonality:**
- `is_holiday_season` - Major holiday period
- `is_tax_season` - Tax filing season
- `is_festival_season` - Festival/celebration period
- `is_school_season` - School fee payment period

---

### Category 7: Treatment Response Features

**Historical Treatment Response:**
- `legal_notice_count` - Legal notices sent
- `legal_response_rate` - Response to legal actions
- `settlement_offer_count` - Settlement offers made
- `settlement_acceptance_rate` - Settlement acceptance rate
- `restructure_count` - Restructuring attempts
- `restructure_success_rate` - Restructuring success rate
- `debt_sale_considered` - Flagged for debt sale

**Channel Effectiveness:**
- `best_performing_channel` - Most effective channel
- `best_performing_time` - Most effective contact time
- `best_performing_day` - Most effective contact day
- `channel_fatigue_score` - Over-contact fatigue (0-1)

**Treatment History:**
- `previous_treatment_path` - Last treatment applied
- `treatment_success_count` - Successful treatments
- `treatment_failure_count` - Failed treatments
- `avg_time_to_treatment_response` - Response time (days)
- `preferred_resolution_method` - Historical preference

---

### Category 8: External & Macro Features

**Economic Indicators:**
- `regional_unemployment_rate` - Local unemployment rate
- `regional_gdp_growth` - Regional economic growth
- `industry_health_index` - Customer's industry health
- `inflation_rate` - Current inflation rate

**Geographic Features:**
- `customer_city` - City of residence
- `customer_state` - State of residence
- `city_tier` - Tier 1/2/3 city classification
- `urban_rural_flag` - Urban vs rural

**Portfolio Features:**
- `portfolio_vintage` - Account vintage cohort
- `portfolio_segment` - Portfolio segment
- `industry_sector` - Customer industry
- `employment_type` - Salaried/self-employed/business

---

### Category 9: Derived Risk Features

**Propensity Scores:**
- `propensity_to_pay_7d` - Likelihood to pay in 7 days
- `propensity_to_pay_30d` - Likelihood to pay in 30 days
- `propensity_to_settle` - Likelihood to accept settlement
- `propensity_to_cure` - Likelihood to fully cure
- `propensity_to_contact_response` - Likelihood to respond

**Expected Values:**
- `expected_payment_amount` - Expected payment amount
- `expected_recovery_rate` - Expected recovery %
- `expected_time_to_cure` - Expected days to cure
- `expected_ltv` - Expected lifetime value

**Composite Scores:**
- `overall_risk_score` - Combined risk score (0-100)
- `payment_capacity_score` - Ability to pay (0-100)
- `willingness_to_pay_score` - Intent to pay (0-100)
- `engagement_quality_score` - Contact quality (0-100)
- `priority_score` - Collection priority (0-100)

---

## 🎯 Critical Features for NBA (Top 30)

**Must-Have Features** for effective NBA modeling:

### Delinquency (5 features)
1. `current_dpd` - Days past due
2. `max_dpd_ever` - Worst delinquency
3. `dpd_trend_30d` - Delinquency direction
4. `delinquency_bucket` - Current bucket
5. `time_in_current_bucket` - Bucket duration

### Payment Behavior (8 features)
6. `days_since_last_payment` - Payment recency
7. `payment_count_30d` - Recent payment frequency
8. `avg_payment_amount_90d` - Average payment
9. `payment_regularity_score` - Payment consistency
10. `promise_kept_ratio` - Promise reliability
11. `partial_payment_ratio` - Partial payment tendency
12. `early_payment_ratio` - On-time payment history
13. `total_paid_amount` - Cumulative recovery

### Contact & Response (5 features)
14. `contact_success_rate` - Reachability
15. `days_since_successful_contact` - Contact recency
16. `preferred_channel` - Best contact method
17. `engagement_score` - Interaction quality
18. `channel_response_rate_best` - Best channel performance

### Financial Health (7 features)
19. `otb_utilization_ratio` - Credit utilization
20. `estimated_monthly_income` - Income level
21. `debt_to_income_ratio` - Debt burden
22. `deposit_regularity_score` - Income stability
23. `free_cash_flow_estimate` - Liquidity
24. `emi_to_income_ratio` - EMI burden
25. `current_utilization` - Product utilization

### Temporal (3 features)
26. `days_to_next_salary` - Payday proximity
27. `is_month_start` - Month timing
28. `is_payday_week` - Payday week

### Treatment Response (2 features)
29. `settlement_acceptance_rate` - Settlement propensity
30. `best_performing_channel` - Optimal treatment

---

## 🔧 Feature Engineering Best Practices

### 1. **Window Selection**
- Use multiple time windows: 7d, 14d, 30d, 60d, 90d, 180d
- Include both short-term (recent) and long-term (historical) patterns
- Align windows with business cycles (billing cycles, payroll)

### 2. **Aggregation Methods**
For each window, compute:
- Mean, median (central tendency)
- Std, CV (variability)
- Min, max, range (extremes)
- Percentiles: 25th, 75th (distribution)
- Skewness, kurtosis (shape)
- Trend direction (linear regression slope)

### 3. **Ratio Features**
Create informative ratios:
- Payment amount / Outstanding balance
- Actual payment / Promised payment
- Successful contacts / Total attempts
- On-time payments / Total payments
- Available credit / Credit limit

### 4. **Velocity, Momentum, Volatility**
- **Velocity**: Rate of change (first derivative)
- **Momentum**: Acceleration (second derivative)
- **Volatility**: Standard deviation over time
- Compute across multiple windows for trends

### 5. **Categorical Encoding**
- One-hot encode: `delinquency_bucket`, `product_type`
- Target encode: `customer_city`, `industry_sector`
- Ordinal encode: `city_tier`, risk buckets
- Frequency encode: Rare categories

### 6. **Missing Value Handling**
- `-999` or specific code for "Never had event"
- `-1` for "Not applicable"
- Median/mean imputation for numerical
- "MISSING" category for categorical
- Create `is_missing` flag features

### 7. **Temporal Alignment**
- All features calculated as of "snapshot date"
- Avoid future leakage (no data after snapshot)
- Use proper time-based train/test splits
- Account for seasonality

### 8. **Interaction Features**
High-value interactions:
- `dpd * days_since_last_payment`
- `otb_utilization * payment_velocity`
- `promise_kept_ratio * contact_success_rate`
- `income_estimate * debt_to_income_ratio`

---

## 📋 Schema Requirements

See **NBA_SCHEMA_TEMPLATE.md** for detailed schema specifications including:
- Required fields
- Optional but recommended fields
- Data types and formats
- Example records
- SQL query templates
- Validation rules

---

## 💡 Next Steps

1. **Review this guide** to understand feature requirements
2. **Check NBA_SCHEMA_TEMPLATE.md** for data schema
3. **Share your current data schema** for gap analysis
4. **Prioritize features** based on data availability
5. **Start with critical features** (Top 30 list above)
6. **Iteratively add** more features as data permits

---

## 📞 Questions?

- What data sources do you have available?
- What's your current data granularity (daily/monthly)?
- How far back is your historical data?
- What are your primary business objectives (recovery rate, cost reduction, customer retention)?

**Ready to share your schema?** → See NBA_SCHEMA_TEMPLATE.md for format

---

*Last updated: 2026-02-11*  
*Feature count: 150+ across 9 categories*
