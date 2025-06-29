import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.image as mpimg
import matplotlib.dates as mdates

print("Script started")

# Ensure output directory exists
os.makedirs('Graphs', exist_ok=True)

# Use native datetime for date arithmetic
today_plus_15 = datetime.now() + timedelta(days=15)
right_limit = today_plus_15 + timedelta(days=30)

# Timeline data (BASt on top, OWM on bottom)
timelines = [
    {
        'label': 'BASt',
        'start': pd.Timestamp('2003-01-01'),
        'end': pd.Timestamp('2023-12-31'),
        'color': '#2c6c88',
    },
    {
        'label': 'OpenWeatherMap API',
        'start': pd.Timestamp('2020-11-27'),
        'end': pd.Timestamp(today_plus_15),
        'color': '#eb6e4b',
    },
]

fig, ax = plt.subplots(figsize=(12, 3))

# Plot each timeline as a horizontal bar (BASt is index 0, OWM is index 1)
bar_positions = [0.65, 0.4]  # Closer together
bar_height = 0.18
bar_edgecolors = ['#1a3a47', '#a13d1d']  # Darker shades for BASt and OWM
for i, tl in enumerate(timelines):
    ax.barh(
        y=bar_positions[i],
        width=(tl['end'] - tl['start']).days,
        left=tl['start'],
        height=bar_height,
        color=tl['color'],
        edgecolor=bar_edgecolors[i],
        linewidth=2,
        alpha=0.95
    )

# Set y-ticks and labels (flipped)
ax.set_yticks(bar_positions)
ax.set_yticklabels([timelines[0]['label'], timelines[1]['label']], fontsize=12)

# Set x-ticks only at 2010-01-01 (labeled '2003-01-01'), 2020-11-27, 2023-12-31, and today+15d
xticks = [pd.Timestamp('2010-01-01'), timelines[1]['start'], timelines[0]['end'], pd.Timestamp(today_plus_15)]
def safe_strftime(ts):
    return ts.strftime('%Y-%m-%d') if not pd.isna(ts) else ''
xticklabels = ['2003-01-01', safe_strftime(timelines[1]['start']), safe_strftime(timelines[0]['end']), 'forecast']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=0, fontsize=11)

# Set x-limits with a little padding
ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2025-07-12'))

# Draw dashed lines to indicate the overlap between BASt and OWM (draw last so they're on top)
for x in [timelines[1]['start'], timelines[0]['end']]:
    ax.axvline(x=x, color='#888888', linestyle='--', linewidth=1.5, zorder=10)

# Remove gridlines
ax.grid(False)

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#888888')
ax.tick_params(axis='y', length=0)

# Title and legend
ax.set_title('BASt vs. OpenWeatherMap API Data Availability Timelines', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('Graphs/bast_owm_timelines.png', dpi=150)

print("Script finished") 