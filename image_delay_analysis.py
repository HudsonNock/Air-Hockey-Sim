import numpy as np

def main():
    times = np.load("new_data/camera_delay_times.npy")

    # Analyze timing
    avg_interval = np.mean(times)
    std_dev = np.std(times)
    min_interval = np.min(times)
    max_interval = np.max(times)
    
    print(f"\n=== Optimized Timing Analysis ===")
    print(f"Average interval: {avg_interval:.4f}ms")
    print(f"Standard deviation: {std_dev:.4f}ms")
    print(f"Min interval: {min_interval:.4f}ms")
    print(f"Max interval: {max_interval:.4f}ms")
    print(f"Jitter range: {max_interval - min_interval:.4f}ms")
    print(f"Actual FPS: {1000/avg_interval:.2f}")
   
    
    # Percentile analysis
    percentiles = [50, 90, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(times, p)
        print(f"{p}th percentile: {val:.4f}ms")
    #119.96 Hz
    # Distribution analysis
    ranges = []
    for i in range(90):
        ranges.append((i/10+13, 13.1 + i/10))
    ranges.append((22,30))
    for min_r, max_r in ranges:
        count = np.sum((times >= min_r) & (times < max_r))
        percentage = (count / len(times)) * 100
        print(f"{min_r:.1f}-{max_r:.1f}ms: {count} frames ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
