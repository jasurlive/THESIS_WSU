# Inverter Short Circuit Faults

Same procedure as for Earth faults.

## What is role of Auxillary converter in electric locomtives?

It provides power to auxillary machines which helps in cooling the system.
Like there are around 17 machines which are supplied power from this converter.
Like traction motor blower to cool the traction motors, oil cooling blower to cool the radiator of transformer, cooling pump for the traction converter coolant, compressor to supply air for pneumatic system etc.

## Primary reasons For Inverter short circuit faults are:

- Short circuit or puncture of IGBT in the Inverter modules.
- Flash in the inverter circuit, earthing of Inverter components due to high current .
- In this case both earth fault and Inverter short circuit will be recorded.

## Auxillary Converter Parameters

- 2 x 130 kVA auxillary converter,
- Nominal Voltage: 830 VAC, 1 Phase
- Frequency: 50 Hz
- Output Data:
  - Power: 2 x 130 kVA at 0.8 PF
  - AC output: 3 x 415 VAC, 50 Hz, Sine wave, VVVF
  - DC Output: 2.2 kW, 110 VDC, 20 A
- Control System:
  - Communication Bus: CAN/Ethernet/MVB
  - Service Signal Connector: RS485

## Parameters Which can be relevant to Inverter faults are:

- There are 3 auxillary converters: Aux1, Aux2,Aux3.
- Aux1 works at fixed frequency 50 Hz.
- Aux2 works at 2 different frequencies 47Hz, 50 Hz depending upon temeprature of traction motors.
- Aux3 works at 2 different frequencies.
- Input voltages R,Y,B of all three converters
- Input current (phase) of all 3 converters.
- Operating frequencies
- Temperature of all three Inverter modules.
- Output RYB voltages and currents for each aux converter.

# Preliminary Data Analysis

Upload both before and after fault .csv (comma separated values) files in Anaconda, VisualStudio, Colab.

### Analyse the data types of each column in both.

### Convert the timestamp column to datetime format.

### For categorical columns use the get_dummies function to convert it to a numeric column.

### Eliminate all columns whose value is the same for all 10 instances.

### Add a column 'fault_label' to differentiate between before and after fault cases.

### Combine both the data frames of after and before faults.

Now the combined data frame is ready for further investigation.
Use group by function to get mean of each column for label_label=0 and fault_label=1. For example,
mean_comparison= combined_data.groupby( 'fault_label')

compare the mean values for each column before fault and after fault to find all columns whose values have changed after fault.
Save all the columns whose values have changed after fault.

# Data Types

### Numeric Types: int64, float64

### Object Type: object

### Categorical Type: category

### Datetime Types: datetime64, timedelta64

### Boolean Type: bool

### Complex Numbers: complex
