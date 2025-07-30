import matplotlib.pyplot as plt

# Data
categories = ['Good', 'Moderate', 'Failure']
values = [31, 5, 14]
colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(categories, values, color=colors, width=0.5, edgecolor='black')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height}', ha='center', va='bottom', fontsize=10)

# Formatting
plt.title('3D Model Generation Results from 50 Prompts', fontsize=12)
plt.ylabel('Number of Prompts', fontsize=11)
plt.ylim(0, max(values) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save if needed
# plt.savefig("3d_generation_results.png", dpi=300)

plt.show()
