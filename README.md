# Inverter Short Circuit Faults

# Example output

![bargraph_result](https://github.com/user-attachments/assets/e824a9ab-be4e-4ab0-8f6c-ef8d5ae20a69)

---

## What is the Role of the Auxiliary Converter in Electric Locomotives?

The auxiliary converter provides power to auxiliary machines, which help in cooling and supporting the locomotive’s electrical systems. Around 17 machines receive power from this converter, including:

- Traction motor blowers for cooling traction motors
- Oil cooling blowers for cooling the transformer's radiator
- Cooling pumps for the traction converter coolant
- Compressors for supplying air to the pneumatic system

---

## Primary Reasons for Inverter Short Circuit Faults

- Short circuit or puncture of IGBTs in the inverter modules
- Flashover in the inverter circuit or earthing of inverter components due to high current
- Both Earth Fault and Inverter Short Circuit Fault are recorded in such cases

---

## Auxiliary Converter Parameters

- **Configuration:** 2 × 130 kVA auxiliary converters
- **Nominal Voltage:** 830 VAC, Single Phase
- **Frequency:** 50 Hz

### Output Data

- **Power:** 2 × 130 kVA at 0.8 Power Factor
- **AC Output:** 3 × 415 VAC, 50 Hz, Sine wave, VVVF
- **DC Output:** 2.2 kW, 110 VDC, 20 A

### Control System

- **Communication Bus:** CAN / Ethernet / MVB
- **Service Signal Connector:** RS485

---

## Parameters Relevant to Inverter Faults

- There are three auxiliary converters: **Aux1**, **Aux2**, and **Aux3**
- **Aux1:** Operates at a fixed frequency of 50 Hz
- **Aux2 & Aux3:** Operate at dual frequencies (e.g., 47 Hz and 50 Hz), depending on the traction motor temperature
- **Monitoring Parameters:**
  - Input voltages (R, Y, B phases) for each converter
  - Input current (per phase) for each converter
  - Operating frequency of each converter
  - Temperature of all three inverter modules
  - Output RYB voltages and currents for each auxiliary converter

---

# Preliminary Data Analysis (Excel-Based Approach)

This analysis directly reads Excel `.xlsx` files using **Pandas**.

### Step-by-Step Instructions:

1. Save your data files (before and after fault) in `.xlsx` format (e.g., `before_fault.xlsx`, `after_fault.xlsx`).

2. Load the Excel files into Python using `pandas.read_excel()`:

   ```python
   import pandas as pd

   before_df = pd.read_excel("before_fault.xlsx")
   after_df = pd.read_excel("after_fault.xlsx")
   ```

3. **Inspect data types** using:

   ```python
   print(before_df.dtypes)
   ```

4. Convert any **timestamp columns** to `datetime` format:

   ```python
   before_df["timestamp"] = pd.to_datetime(before_df["timestamp"])
   after_df["timestamp"] = pd.to_datetime(after_df["timestamp"])
   ```

5. Convert **categorical columns** to numeric using one-hot encoding:

   ```python
   before_df = pd.get_dummies(before_df)
   after_df = pd.get_dummies(after_df)
   ```

6. Remove columns that are **constant across all rows**:

   ```python
   before_df = before_df.loc[:, (before_df != before_df.iloc[0]).any()]
   after_df = after_df.loc[:, (after_df != after_df.iloc[0]).any()]
   ```

7. Add a `fault_label` column:

   ```python
   before_df["fault_label"] = 0
   after_df["fault_label"] = 1
   ```

8. Combine the data:

   ```python
   combined_data = pd.concat([before_df, after_df], ignore_index=True)
   ```

---

## Comparative Analysis

Use the `groupby()` function to compute the **mean of each column** based on `fault_label`.

```python
mean_comparison = combined_data.groupby("fault_label").mean()
```

Then compare the mean values of all columns between:

- `fault_label = 0` → Before Fault
- `fault_label = 1` → After Fault

Identify which parameters show significant changes after the fault. Save or log all such columns for further investigation.

---

# Data Types in Pandas

| Type               | Description                     |
| ------------------ | ------------------------------- |
| `int64`, `float64` | Numeric types                   |
| `object`           | Text/categorical values         |
| `category`         | Categorical type (efficient)    |
| `datetime64`       | Date and time values            |
| `timedelta64`      | Time durations                  |
| `bool`             | Boolean values (`True`/`False`) |
| `complex`          | Complex numbers                 |
