import pandas as pd
import matplotlib.pyplot as plt
# 3D plot of time, load and displacement
df = pd.read_csv('Specimen_RawData_1.csv')
# drop the first row which contains units
df = df.drop(index=0)
# plot only after every 10th point to reduce clutter
df = df.iloc[::10]

fig = plt.figure(figsize=(10, 7))
plt.plot(df['Load'], df['Extension'], label='Load vs Displacement')
plt.xlabel('Load (N)')
plt.ylabel('Extension (mm)')
plt.title('Plot of Load vs Displacement')
plt.legend()
plt.savefig('3d_plot.png', dpi=300)