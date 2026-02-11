# Collections NBA Data Schema Template

> **Reference schema for Collections NBA feature engineering and modeling**

---

## 📋 Quick Reference

**Minimum Required Tables:**
1. Account Snapshot (current state)
2. Payment History (transactional)
3. Contact History (communication logs)
4. Promise History (PTP tracking)

**Optional but Recommended:**
5. Transaction History (spending behavior)
6. Deposit History (income proxy)
7. Treatment History (collection actions)

---

## 🗂️ Schema Overview

### Table 1: Account Snapshot (Current State)

**Purpose:** Current account status as of snapshot date  
**Granularity:** One record per account per snapshot date  
**Update Frequency:** Daily or monthly

#### Required Fields

| Field Name | Data Type | Description | Example | Constraint |
|------------|-----------|-------------|---------|------------|
| `account_id` | VARCHAR(50) | Unique account identifier | "ACC123456" | NOT NULL, PK |
| `snapshot_date` | DATE | As-of date for snapshot | "2026-02-11" | NOT NULL, PK |
| `customer_id` | VARCHAR(50) | Customer identifier | "CUST78910" | NOT NULL |
| `product_type` | VARCHAR(50) | Product category | "CREDIT_CARD", "PERSONAL_LOAN" | NOT NULL |
| `account_open_date` | DATE | Account opening date | "2023-05-15" | NOT NULL |
| `current_balance` | DECIMAL(15,2) | Total outstanding balance | 45000.00 | >= 0 |
| `principal_balance` | DECIMAL(15,2) | Principal amount | 40000.00 | >= 0 |
| `interest_balance` | DECIMAL(15,2) | Interest amount | 4500.00 | >= 0 |
| `fees_balance` | DECIMAL(15,2) | Fees and charges | 500.00 | >= 0 |
| `current_dpd` | INTEGER | Days past due | 45 | >= 0 |
| `delinquency_bucket` | VARCHAR(20) | Delinquency category | "31-60", "61-90" | NOT NULL |
| `credit_limit` | DECIMAL(15,2) | Credit limit (revolving) | 100000.00 | > 0 or NULL |
| `original_loan_amount` | DECIMAL(15,2) | Original disbursement | 50000.00 | > 0 or NULL |
| `emi_amount` | DECIMAL(15,2) | Monthly EMI (installment) | 5000.00 | >= 0 or NULL |
| `remaining_tenure` | INTEGER | Months to maturity | 36 | >= 0 or NULL |

#### Recommended Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `interest_rate` | DECIMAL(5,2) | Annual interest rate % | 18.50 |
| `max_dpd_ever` | INTEGER | Maximum DPD in history | 90 |
| `cycle_dpd` | INTEGER | DPD at last cycle | 30 |
| `bucket_entry_date` | DATE | Date entered current bucket | "2026-01-15" |
| `last_payment_date` | DATE | Most recent payment date | "2026-01-20" |
| `last_payment_amount` | DECIMAL(15,2) | Last payment amount | 2000.00 |
| `total_payments_received` | DECIMAL(15,2) | Lifetime payments | 25000.00 |
| `write_off_flag` | BOOLEAN | Written off indicator | 0 or 1 |
| `legal_status` | VARCHAR(20) | Legal action status | "NONE", "NOTICE_SENT" |

---

### Table 2: Payment History

**Purpose:** All payment transactions  
**Granularity:** One record per payment  
**Update Frequency:** Real-time or daily

#### Required Fields

| Field Name | Data Type | Description | Example | Constraint |
|------------|-----------|-------------|---------|------------|
| `payment_id` | VARCHAR(50) | Unique payment ID | "PAY123456" | NOT NULL, PK |
| `account_id` | VARCHAR(50) | Account identifier | "ACC123456" | NOT NULL, FK |
| `payment_date` | DATE | Payment date | "2026-02-10" | NOT NULL |
| `payment_amount` | DECIMAL(15,2) | Payment amount | 5000.00 | > 0 |
| `payment_method` | VARCHAR(30) | Payment method | "ONLINE", "BRANCH", "AUTO_DEBIT" | NOT NULL |
| `payment_status` | VARCHAR(20) | Payment status | "SUCCESS", "FAILED", "PENDING" | NOT NULL |

#### Recommended Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `due_date` | DATE | Associated due date | "2026-02-05" |
| `payment_category` | VARCHAR(20) | Full/partial/min | "PARTIAL" |
| `dpd_at_payment` | INTEGER | DPD when paid | 35 |
| `balance_before` | DECIMAL(15,2) | Balance before payment | 45000.00 |
| `balance_after` | DECIMAL(15,2) | Balance after payment | 40000.00 |
| `payment_channel` | VARCHAR(30) | Collection channel | "SELF", "REMINDER", "CALL" |
| `promise_id` | VARCHAR(50) | Related promise (if any) | "PRM123" |

---

### Table 3: Contact History

**Purpose:** All contact attempts and outcomes  
**Granularity:** One record per contact attempt  
**Update Frequency:** Real-time or daily

#### Required Fields

| Field Name | Data Type | Description | Example | Constraint |
|------------|-----------|-------------|---------|------------|
| `contact_id` | VARCHAR(50) | Unique contact ID | "CNT123456" | NOT NULL, PK |
| `account_id` | VARCHAR(50) | Account identifier | "ACC123456" | NOT NULL, FK |
| `contact_date` | TIMESTAMP | Contact date/time | "2026-02-10 14:30:00" | NOT NULL |
| `contact_channel` | VARCHAR(20) | Contact channel | "PHONE", "SMS", "EMAIL", "WHATSAPP" | NOT NULL |
| `contact_outcome` | VARCHAR(30) | Outcome | "CONNECTED", "NO_ANSWER", "WRONG_NUMBER" | NOT NULL |

#### Recommended Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `contact_type` | VARCHAR(20) | Inbound/outbound | "OUTBOUND" |
| `right_party_contact` | BOOLEAN | Right person reached | 1 |
| `call_duration_sec` | INTEGER | Call duration (seconds) | 180 |
| `agent_id` | VARCHAR(30) | Agent identifier | "AGENT001" |
| `disposition_code` | VARCHAR(50) | Detailed outcome | "PTP_GIVEN", "DISPUTE_RAISED" |
| `contact_notes` | TEXT | Contact notes | "Customer agreed to pay by Friday" |
| `sentiment` | VARCHAR(20) | Interaction tone | "COOPERATIVE", "HOSTILE", "NEUTRAL" |
| `next_action` | VARCHAR(50) | Follow-up action | "CALLBACK_SCHEDULED" |

---

### Table 4: Promise History

**Purpose:** Promise-to-Pay (PTP) tracking  
**Granularity:** One record per promise  
**Update Frequency:** Real-time or daily

#### Required Fields

| Field Name | Data Type | Description | Example | Constraint |
|------------|-----------|-------------|---------|------------|
| `promise_id` | VARCHAR(50) | Unique promise ID | "PRM123456" | NOT NULL, PK |
| `account_id` | VARCHAR(50) | Account identifier | "ACC123456" | NOT NULL, FK |
| `promise_date` | DATE | Promise made date | "2026-02-08" | NOT NULL |
| `promise_amount` | DECIMAL(15,2) | Promised amount | 10000.00 | > 0 |
| `promise_due_date` | DATE | Promise due date | "2026-02-15" | NOT NULL |
| `promise_status` | VARCHAR(20) | Status | "KEPT", "BROKEN", "PENDING" | NOT NULL |

#### Recommended Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `actual_payment_date` | DATE | Actual payment date | "2026-02-14" |
| `actual_payment_amount` | DECIMAL(15,2) | Actual paid amount | 10000.00 |
| `contact_id` | VARCHAR(50) | Related contact | "CNT123456" |
| `promise_channel` | VARCHAR(20) | Channel of promise | "PHONE" |
| `days_to_fulfillment` | INTEGER | Days from promise to payment | 6 |
| `fulfillment_percentage` | DECIMAL(5,2) | % of promise fulfilled | 100.00 |

---

### Table 5: Transaction History (Optional)

**Purpose:** Spending and transaction behavior  
**Granularity:** One record per transaction  
**Update Frequency:** Real-time or daily

#### Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `transaction_id` | VARCHAR(50) | Unique transaction ID | "TXN123456" |
| `account_id` | VARCHAR(50) | Account identifier | "ACC123456" |
| `transaction_date` | TIMESTAMP | Transaction timestamp | "2026-02-10 09:15:00" |
| `transaction_amount` | DECIMAL(15,2) | Transaction amount | 2500.00 |
| `transaction_type` | VARCHAR(20) | Type | "PURCHASE", "CASH_ADVANCE" |
| `merchant_category` | VARCHAR(50) | MCC category | "GROCERY", "FUEL", "DINING" |
| `merchant_name` | VARCHAR(100) | Merchant name | "SUPERMART XYZ" |

---

### Table 6: Deposit History (Optional - Income Proxy)

**Purpose:** Deposit patterns as income indicator  
**Granularity:** One record per deposit  
**Update Frequency:** Daily

#### Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `deposit_id` | VARCHAR(50) | Unique deposit ID | "DEP123456" |
| `customer_id` | VARCHAR(50) | Customer identifier | "CUST78910" |
| `deposit_date` | DATE | Deposit date | "2026-02-01" |
| `deposit_amount` | DECIMAL(15,2) | Deposit amount | 50000.00 |
| `deposit_source` | VARCHAR(50) | Source | "SALARY", "TRANSFER", "CASH" |
| `account_type` | VARCHAR(20) | Account type | "SAVINGS", "CURRENT" |

---

### Table 7: Treatment History (Optional)

**Purpose:** Collection treatment history  
**Granularity:** One record per treatment action  
**Update Frequency:** Real-time or daily

#### Fields

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `treatment_id` | VARCHAR(50) | Unique treatment ID | "TRT123456" |
| `account_id` | VARCHAR(50) | Account identifier | "ACC123456" |
| `treatment_date` | DATE | Treatment date | "2026-01-20" |
| `treatment_type` | VARCHAR(30) | Treatment type | "LEGAL_NOTICE", "SETTLEMENT_OFFER" |
| `treatment_outcome` | VARCHAR(30) | Outcome | "ACCEPTED", "REJECTED", "NO_RESPONSE" |
| `settlement_amount` | DECIMAL(15,2) | Settlement amount (if applicable) | 30000.00 |
| `treatment_cost` | DECIMAL(10,2) | Cost of treatment | 500.00 |

---

## 📊 Example Data Samples

### Account Snapshot Example

```sql
SELECT 
    'ACC123456' as account_id,
    '2026-02-11' as snapshot_date,
    'CUST78910' as customer_id,
    'CREDIT_CARD' as product_type,
    '2023-05-15' as account_open_date,
    45000.00 as current_balance,
    40000.00 as principal_balance,
    4500.00 as interest_balance,
    500.00 as fees_balance,
    45 as current_dpd,
    '31-60' as delinquency_bucket,
    100000.00 as credit_limit,
    18.50 as interest_rate,
    90 as max_dpd_ever,
    '2026-01-15' as bucket_entry_date,
    '2026-01-20' as last_payment_date,
    2000.00 as last_payment_amount;
```

### Payment History Example

```sql
SELECT 
    'PAY123456' as payment_id,
    'ACC123456' as account_id,
    '2026-02-10' as payment_date,
    5000.00 as payment_amount,
    'ONLINE' as payment_method,
    'SUCCESS' as payment_status,
    '2026-02-05' as due_date,
    'PARTIAL' as payment_category,
    35 as dpd_at_payment,
    45000.00 as balance_before,
    40000.00 as balance_after;
```

---

## 🔍 Data Quality Requirements

### Completeness

**Required Fields:**
- < 5% missing values in required fields
- 0% missing in: account_id, snapshot_date, current_balance, current_dpd

**Optional Fields:**
- Flag records with missing data
- Use appropriate defaults or imputation

### Consistency

**Cross-Table Checks:**
- All account_ids in payment history exist in account snapshot
- Payment dates <= snapshot_date
- DPD calculation consistent with payment dates
- Balance updates match payment amounts

**Logical Checks:**
- current_balance >= 0
- current_dpd >= 0
- credit_limit >= current_balance (for revolving)
- payment_amount > 0
- promise_amount > 0

### Accuracy

**Date Validation:**
- All dates in valid format (YYYY-MM-DD)
- No future dates (except promise_due_date)
- Chronological order maintained

**Amount Validation:**
- Amounts in valid decimal format
- No negative values where not applicable
- Reasonable ranges (no outliers beyond 3 sigma)

---

## 🛠️ SQL Query Templates

### Generate Account Features

```sql
-- Account snapshot with basic features
SELECT 
    a.account_id,
    a.snapshot_date,
    a.current_dpd,
    a.current_balance,
    a.delinquency_bucket,
    DATEDIFF(a.snapshot_date, a.account_open_date) as account_age_days,
    
    -- Payment features (last 30 days)
    COUNT(DISTINCT p.payment_id) as payment_count_30d,
    COALESCE(SUM(p.payment_amount), 0) as total_paid_30d,
    COALESCE(AVG(p.payment_amount), 0) as avg_payment_30d,
    DATEDIFF(a.snapshot_date, MAX(p.payment_date)) as days_since_last_payment,
    
    -- Contact features (last 30 days)
    COUNT(DISTINCT c.contact_id) as contact_count_30d,
    SUM(CASE WHEN c.contact_outcome = 'CONNECTED' THEN 1 ELSE 0 END) as successful_contact_30d,
    
    -- Promise features
    COUNT(DISTINCT pr.promise_id) as promise_count,
    SUM(CASE WHEN pr.promise_status = 'KEPT' THEN 1 ELSE 0 END) as promise_kept_count,
    SUM(CASE WHEN pr.promise_status = 'BROKEN' THEN 1 ELSE 0 END) as promise_broken_count
    
FROM account_snapshot a
LEFT JOIN payment_history p 
    ON a.account_id = p.account_id 
    AND p.payment_date BETWEEN DATEADD(DAY, -30, a.snapshot_date) AND a.snapshot_date
    AND p.payment_status = 'SUCCESS'
LEFT JOIN contact_history c
    ON a.account_id = c.account_id
    AND c.contact_date BETWEEN DATEADD(DAY, -30, a.snapshot_date) AND a.snapshot_date
LEFT JOIN promise_history pr
    ON a.account_id = pr.account_id
    AND pr.promise_date <= a.snapshot_date
GROUP BY 
    a.account_id, 
    a.snapshot_date,
    a.current_dpd,
    a.current_balance,
    a.delinquency_bucket,
    a.account_open_date;
```

### Generate Behavioral Features

```sql
-- Velocity and volatility features
WITH payment_metrics AS (
    SELECT 
        account_id,
        -- 30-day metrics
        COUNT(*) as payment_count_30d,
        AVG(payment_amount) as avg_payment_30d,
        STDDEV(payment_amount) as std_payment_30d,
        MIN(payment_amount) as min_payment_30d,
        MAX(payment_amount) as max_payment_30d
    FROM payment_history
    WHERE payment_date BETWEEN DATEADD(DAY, -30, CURRENT_DATE) AND CURRENT_DATE
        AND payment_status = 'SUCCESS'
    GROUP BY account_id
)
SELECT 
    account_id,
    payment_count_30d,
    avg_payment_30d,
    std_payment_30d,
    CASE 
        WHEN avg_payment_30d > 0 THEN std_payment_30d / avg_payment_30d 
        ELSE NULL 
    END as payment_cv_30d,  -- Coefficient of variation
    max_payment_30d - min_payment_30d as payment_range_30d
FROM payment_metrics;
```

---

## ✅ Validation Checklist

Before sharing your schema, verify:

- [ ] All required tables exist
- [ ] Primary keys defined on all tables
- [ ] Foreign keys properly linked
- [ ] Date fields in correct format
- [ ] Amount fields as DECIMAL (not FLOAT)
- [ ] No negative values where inappropriate
- [ ] NULL handling strategy defined
- [ ] Historical data depth >= 6 months
- [ ] Daily/monthly granularity specified
- [ ] Sample data provided for review

---

## 📤 How to Share Your Schema

**Please provide:**

1. **Table Definitions** - DDL scripts or ER diagram
2. **Sample Data** - 10-20 rows per table
3. **Data Dictionary** - Field descriptions
4. **Availability** - Which fields are currently populated
5. **Historical Depth** - How far back does data go
6. **Update Frequency** - Real-time, daily, monthly?
7. **Data Volume** - Number of accounts, transactions

**Format Options:**
- SQL DDL scripts
- Excel/CSV with schema
- ER diagram (image/PDF)
- Data profiling report

---

## 🎯 Gap Analysis Process

Once you share your schema, I'll help with:

1. **Field Mapping** - Map your fields to NBA features
2. **Gap Identification** - Identify missing critical features
3. **Workaround Strategies** - Alternative feature engineering
4. **Priority Features** - Must-have vs nice-to-have
5. **SQL Queries** - Custom feature extraction queries
6. **Feature Engineering Plan** - Step-by-step implementation

---

## 💡 Next Steps

1. **Review** NBA_FEATURES_GUIDE.md for feature requirements
2. **Prepare** your current schema documentation
3. **Share** schema following guidelines above
4. **Collaborate** on gap analysis and feature engineering plan
5. **Implement** priority features first
6. **Iterate** based on model performance

---

**Ready to share your schema?** Include:
- Table structures
- Sample data
- Data dictionary
- Known limitations

---

*Last updated: 2026-02-11*  
*Template version: 1.0*
