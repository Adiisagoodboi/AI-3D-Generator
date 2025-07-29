import matplotlib.pyplot as plt

# Generation times for 6 runs at each karras_step
times_32 = [643, 645, 653, 651, 650, 648]
times_50 = [700, 710, 720, 715, 712, 720]
times_64 = [765, 770, 775, 780, 785, 778]

# Run numbers (1 through 6)
runs = [1, 2, 3, 4, 5, 6]

# Plot the lines
plt.figure(figsize=(10, 6))
plt.plot(runs, times_32, marker='o', label='Karras Step 32', color='blue')
plt.plot(runs, times_50, marker='o', label='Karras Step 50', color='green')
plt.plot(runs, times_64, marker='o', label='Karras Step 64', color='red')

# Chart formatting
plt.title('Generation Time per Run for Different Karras Steps')
plt.xlabel('Run Number')
plt.ylabel('Generation Time (seconds)')
plt.xticks(runs)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
