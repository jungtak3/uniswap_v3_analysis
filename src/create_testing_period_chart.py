"""
Create Testing Period Close Price Chart
Visualizes market conditions during backtesting period for Uniswap V3 strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

def load_and_split_data(data_path=None, test_size=0.3):
    """Load and split data exactly like the strategy framework"""
    
    if data_path is None:
        # Try multiple possible paths for the data file
        possible_paths = [
            'ETHUSDC_20181215_20250430.csv',
            '../ETHUSDC_20181215_20250430.csv',
            '../downloaded_klines/ETHUSDC_20181215_20250430.csv',
            '../../downloaded_klines/ETHUSDC_20181215_20250430.csv'
        ]
        
        data_path = None
        for path in possible_paths:
            import os
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"Data file not found. Please ensure ETHUSDC_20181215_20250430.csv is in one of these locations:\n" +
                "\n".join(f"  - {path}" for path in possible_paths)
            )
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Convert timestamps (same as strategy framework)
    data['timestamp'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('timestamp', inplace=True)
    
    # Calculate returns and indicators (same as framework)
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Rolling volatility (20-day window)
    data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(365)
    
    # Price momentum indicators
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    
    # Remove NaN values
    data.dropna(inplace=True)
    
    # Split data (same logic as framework)
    split_index = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    
    return data, train_data, test_data

def create_testing_period_chart():
    """Create comprehensive testing period visualization"""
    
    print("Loading data and creating testing period chart...")
    
    try:
        full_data, train_data, test_data = load_and_split_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure ETHUSDC_20181215_20250430.csv is in the current directory or downloaded_klines/ folder")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Testing Period Market Analysis for Uniswap V3 Strategies', fontsize=16, fontweight='bold')
    
    # 1. Price Chart with Train/Test Split
    ax1 = axes[0]
    
    # Plot full price history
    ax1.plot(train_data.index, train_data['close'], color='blue', alpha=0.7, linewidth=1, label='Training Period')
    ax1.plot(test_data.index, test_data['close'], color='red', alpha=0.8, linewidth=1.5, label='Testing Period')
    
    # Add vertical line at split
    split_date = test_data.index[0]
    ax1.axvline(x=split_date, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Train/Test Split')
    
    # Identify and highlight data gap in early 2023
    from datetime import datetime
    gap_start = datetime(2023, 1, 1)
    gap_end = datetime(2023, 3, 12)
    
    # Add shaded region for data gap
    ax1.axvspan(gap_start, gap_end, alpha=0.3, color='gray', label='Data Gap (No Trading Data)')
    
    # Add circle annotation around the gap period
    from matplotlib.patches import Ellipse
    import matplotlib.dates as mdates
    
    # Convert dates to matplotlib format for circle positioning
    gap_center = gap_start + (gap_end - gap_start) / 2
    gap_center_num = mdates.date2num(gap_center)
    
    # Get price range for circle sizing
    price_range = ax1.get_ylim()
    circle_height = (price_range[1] - price_range[0]) * 0.4
    circle_width = 45  # days in matplotlib date format
    
    # Create circle annotation
    circle = Ellipse((gap_center_num, (price_range[1] + price_range[0]) / 2),
                    circle_width, circle_height,
                    fill=False, edgecolor='red', linewidth=3, linestyle='--',
                    alpha=0.8)
    ax1.add_patch(circle)
    
    # Add text annotation
    ax1.annotate('DATA EMPTY\n(Jan-Mar 2023)',
                xy=(gap_center_num, (price_range[1] + price_range[0]) / 2),
                xytext=(gap_center_num + 100, price_range[1] * 0.9),
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                         edgecolor='red', alpha=0.8))
    
    # Add moving averages for testing period
    ax1.plot(test_data.index, test_data['sma_20'], color='orange', alpha=0.6, linewidth=1, label='SMA 20 (Test)')
    ax1.plot(test_data.index, test_data['sma_50'], color='purple', alpha=0.6, linewidth=1, label='SMA 50 (Test)')
    
    ax1.set_title('ETH/USDC Price: Training vs Testing Periods (Data Gap Highlighted)')
    ax1.set_ylabel('Price (USDC)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # 2. Testing Period Volatility
    ax2 = axes[1]
    
    ax2.plot(test_data.index, test_data['volatility'], color='darkred', linewidth=1.5)
    ax2.fill_between(test_data.index, test_data['volatility'], alpha=0.3, color='red')
    
    # Add volatility statistics
    avg_vol = test_data['volatility'].mean()
    ax2.axhline(y=avg_vol, color='black', linestyle='-', alpha=0.7, label=f'Avg Volatility: {avg_vol:.2f}')
    
    ax2.set_title('Testing Period Volatility (Annualized)')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # 3. Testing Period Returns Distribution
    ax3 = axes[2]
    
    # Daily returns histogram
    test_data['returns'].hist(bins=50, alpha=0.7, color='steelblue', ax=ax3)
    ax3.axvline(x=test_data['returns'].mean(), color='red', linestyle='--', 
                label=f'Mean: {test_data["returns"].mean():.4f}')
    ax3.axvline(x=test_data['returns'].median(), color='green', linestyle='--', 
                label=f'Median: {test_data["returns"].median():.4f}')
    
    ax3.set_title('Testing Period Daily Returns Distribution')
    ax3.set_xlabel('Daily Returns')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    chart_filename = 'uniswap_v3_testing_period_analysis.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TESTING PERIOD MARKET ANALYSIS")
    print("="*80)
    print(f"Testing Period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Duration: {len(test_data)} days")
    print("⚠️  DATA GAP: January 1, 2023 - March 11, 2023 (NO TRADING DATA)")
    print(f"Actual data starts: March 12, 2023")
    print(f"Price Range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"Price Change: {((test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Average Daily Return: {test_data['returns'].mean() * 100:.3f}%")
    print(f"Volatility (Annualized): {test_data['volatility'].mean():.2f}")
    print(f"Max Drawdown: {((test_data['close'] / test_data['close'].expanding().max()) - 1).min() * 100:.2f}%")
    
    # Identify major price movements
    large_moves = test_data[abs(test_data['returns']) > 0.1]  # >10% daily moves
    if len(large_moves) > 0:
        print(f"\nLarge Daily Moves (>10%):")
        for date, row in large_moves.iterrows():
            print(f"  {date.strftime('%Y-%m-%d')}: {row['returns'] * 100:.2f}% (Price: ${row['close']:.2f})")
    
    print(f"\nChart saved as: {chart_filename}")
    print("="*80)

def create_strategy_context_chart():
    """Create additional context for strategy performance analysis"""
    
    try:
        full_data, train_data, test_data = load_and_split_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create market regime analysis
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Calculate price momentum and trends
    test_data['price_change_20d'] = test_data['close'].pct_change(20)
    test_data['trend_strength'] = test_data['sma_20'] / test_data['sma_50'] - 1
    
    # Color code based on market regime
    colors = []
    regimes = []
    for _, row in test_data.iterrows():
        if row['trend_strength'] > 0.02:  # Strong uptrend
            colors.append('green')
            regimes.append('Bull Market')
        elif row['trend_strength'] < -0.02:  # Strong downtrend
            colors.append('red')
            regimes.append('Bear Market')
        else:  # Sideways
            colors.append('gray')
            regimes.append('Sideways')
    
    # Plot price with regime coloring
    for i in range(len(test_data) - 1):
        ax.plot(test_data.index[i:i+2], test_data['close'].iloc[i:i+2], 
                color=colors[i], alpha=0.8, linewidth=2)
    
    ax.set_title('Testing Period: Market Regimes for Uniswap V3 Strategy Analysis')
    ax.set_ylabel('ETH/USDC Price')
    ax.set_xlabel('Date')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Bull Market (SMA20 > SMA50 * 1.02)'),
        Line2D([0], [0], color='red', lw=2, label='Bear Market (SMA20 < SMA50 * 0.98)'),
        Line2D([0], [0], color='gray', lw=2, label='Sideways Market')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    plt.savefig('uniswap_v3_market_regimes.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print regime statistics
    regime_stats = pd.Series(regimes).value_counts()
    print(f"\nMarket Regime Distribution (Testing Period):")
    for regime, count in regime_stats.items():
        percentage = (count / len(regimes)) * 100
        print(f"  {regime}: {count} days ({percentage:.1f}%)")

if __name__ == "__main__":
    print("Creating Uniswap V3 Testing Period Analysis Charts...")
    create_testing_period_chart()
    print("\nCreating Market Regime Analysis...")
    create_strategy_context_chart()
    print("\nAnalysis complete!")