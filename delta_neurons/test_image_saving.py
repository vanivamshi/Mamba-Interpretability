#!/usr/bin/env python3
"""
Test script to verify image saving functionality.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

def test_image_saving():
    """Test that images can be saved correctly."""
    print("🖼️ Testing image saving functionality...")
    
    # Create images directory
    os.makedirs("images", exist_ok=True)
    
    # Test 1: Simple plot
    print("📊 Creating test plot 1...")
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title("Test Plot 1: Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("images/test_plot_1.png", dpi=300, bbox_inches='tight')
    print("✅ Saved test_plot_1.png")
    plt.close()
    
    # Test 2: Bar plot
    print("📊 Creating test plot 2...")
    plt.figure(figsize=(10, 6))
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    plt.bar(categories, values, color='skyblue', alpha=0.7)
    plt.title("Test Plot 2: Bar Chart")
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("images/test_plot_2.png", dpi=300, bbox_inches='tight')
    print("✅ Saved test_plot_2.png")
    plt.close()
    
    # Test 3: Heatmap
    print("📊 Creating test plot 3...")
    plt.figure(figsize=(8, 6))
    data = np.random.randn(10, 10)
    # Ensure square cells by setting proper aspect ratio
    im = plt.imshow(data, cmap='viridis', aspect='equal', interpolation='nearest')
    plt.colorbar(im)
    plt.title("Test Plot 3: Heatmap")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("images/test_plot_3.png", dpi=300, bbox_inches='tight')
    print("✅ Saved test_plot_3.png")
    plt.close()
    
    # Check if files were created
    expected_files = ["test_plot_1.png", "test_plot_2.png", "test_plot_3.png"]
    created_files = []
    
    for filename in expected_files:
        filepath = os.path.join("images", filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"✅ {filename} exists ({file_size:.1f} KB)")
            created_files.append(filename)
        else:
            print(f"❌ {filename} was not created")
    
    # List all files in images directory
    print(f"\n📁 All files in images directory:")
    if os.path.exists("images"):
        all_files = os.listdir("images")
        for i, filename in enumerate(sorted(all_files), 1):
            filepath = os.path.join("images", filename)
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"  {i:2d}. {filename} ({file_size:.1f} KB)")
    
    success_rate = len(created_files) / len(expected_files) * 100
    print(f"\n🎯 Image saving success rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All test images saved successfully!")
        return True
    else:
        print("❌ Some images failed to save")
        return False

if __name__ == "__main__":
    test_image_saving() 