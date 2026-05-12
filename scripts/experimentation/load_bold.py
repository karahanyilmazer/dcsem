# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
df = pd.read_csv("data/sub-karahan_run-01_DMN_timeseries.csv", index_col=0)
print(df.shape)  # expected: (588, 4)
print(df.head())

# %%
plt.figure(figsize=(12, 6))
plt.plot(df["PCC"], label="PCC")
plt.plot(df["mPFC"], label="mPFC")
plt.plot(df["LIPC"], label="LIPC")
plt.plot(df["RIPC"], label="RIPC")
plt.xlabel("Time")
plt.ylabel("PCC")
plt.title("DMN Timeseries")
plt.legend()
plt.show()

# %%
