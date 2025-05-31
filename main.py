#!/usr/bin/env python3
"""
Main execution script for Uniswap V3 Strategy Analysis
Runs both analysis scripts in sequence
"""

import sys
import os
import subprocess
from pathlib import Path

def check_data_file():
    """Check if required data file exists"""
    data_files = [
        'ETHUSDC_20181215_20250430.csv',
        'downloaded_klines/ETHUSDC_20181215_20250430.csv',
        '../ETHUSDC_20181215_20250430.csv',
        '../downloaded_klines/ETHUSDC_20181215_20250430.csv'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✓ Data file found: {data_file}")
            return True
    
    print("❌ Data file not found!")
    print("Please ensure ETHUSDC_20181215_20250430.csv is in one of these locations:")
    for data_file in data_files:
        print(f"  - {data_file}")
    return False

def run_script(script_path, script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    try:
        # Change to src directory for execution
        src_dir = Path(__file__).parent / 'src'
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=src_dir,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully!")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 Uniswap V3 Strategy Analysis - Main Execution Script")
    print("=" * 60)
    
    # Check if data file exists
    if not check_data_file():
        print("\n❌ Cannot proceed without data file. Exiting...")
        sys.exit(1)
    
    # Define scripts to run
    scripts = [
        ('create_testing_period_chart.py', 'Testing Period Chart Generator'),
        ('uniswapv3_strategy_fixed.py', 'Uniswap V3 Strategy Analysis')
    ]
    
    success_count = 0
    
    # Run each script
    for script_file, script_name in scripts:
        if run_script(script_file, script_name):
            success_count += 1
        else:
            print(f"\n⚠️  {script_name} failed - continuing with next script...")
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Scripts completed successfully: {success_count}/{len(scripts)}")
    
    if success_count == len(scripts):
        print("🎉 All scripts completed successfully!")
        print("\nGenerated files:")
        print("  📊 uniswap_v3_testing_period_analysis.png")
        print("  📊 uniswap_v3_market_regimes.png")
        print("  📊 uniswap_v3_strategy_analysis_fixed.png")
        print("  📄 uniswap_v3_strategy_performance_fixed.csv")
    else:
        print("⚠️  Some scripts failed. Check the output above for details.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()