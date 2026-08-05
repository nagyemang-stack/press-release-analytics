"""
Press Release Performance Analytics
====================================
A data-driven case study analyzing press release distribution metrics
across media outlets. Tracks pickup rates, geographic reach, audience
impressions, and ROI correlation.

Author: Caleb Agyemang
Role: PR & Data Analytics Professional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman'],
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

# Colors
NAVY = '#16213E'
TEAL = '#0D9488'
RED = '#C0392B'
AMBER = '#E2A847'
GRAY = '#94A3B8'

# ─── Step 1: Generate Press Release Distribution Data ───
def generate_outlet_data():
    """Generate media outlet performance dataset."""
    np.random.seed(7)
    
    outlets = [
        {'name': 'Bloomberg', 'tier': 1, 'audience': 61000, 'region': 'North America'},
        {'name': 'AP News', 'tier': 1, 'audience': 52000, 'region': 'North America'},
        {'name': 'Reuters', 'tier': 1, 'audience': 35000, 'region': 'Europe'},
        {'name': 'TechCrunch', 'tier': 1, 'audience': 48000, 'region': 'North America'},
        {'name': 'Forbes', 'tier': 2, 'audience': 42000, 'region': 'North America'},
        {'name': 'Business Insider', 'tier': 2, 'audience': 28000, 'region': 'North America'},
        {'name': 'Yahoo Finance', 'tier': 2, 'audience': 18000, 'region': 'North America'},
        {'name': 'MarketWatch', 'tier': 2, 'audience': 15000, 'region': 'North America'},
        {'name': 'BBC News', 'tier': 1, 'audience': 45000, 'region': 'Europe'},
        {'name': 'Financial Times', 'tier': 1, 'audience': 38000, 'region': 'Europe'},
        {'name': 'Nikkei Asia', 'tier': 2, 'audience': 22000, 'region': 'Asia Pacific'},
        {'name': 'South China Morning Post', 'tier': 2, 'audience': 16000, 'region': 'Asia Pacific'},
    ]
    
    data = []
    for outlet in outlets:
        # Pickup probability based on tier
        pickup_prob = np.random.uniform(0.65, 0.98) if outlet['tier'] == 1 else np.random.uniform(0.55, 0.85)
        # Time to coverage (hours) — faster for tier 1
        time_to_pickup = np.random.exponential(48 if outlet['tier'] == 1 else 72)
        # Engagement multiplier based on UTM tracking
        utm_multiplier = np.random.uniform(2.5, 4.2) if np.random.random() > 0.4 else np.random.uniform(0.8, 1.5)
        
        data.append({
            'outlet': outlet['name'],
            'tier': outlet['tier'],
            'audience_reach': outlet['audience'],
            'region': outlet['region'],
            'pickup_rate': round(pickup_prob * 100, 1),
            'was_picked_up': pickup_prob > 0.5,
            'time_to_coverage_hours': round(time_to_pickup, 1),
            'impressions': int(outlet['audience'] * pickup_prob * np.random.uniform(0.8, 1.2)),
            'utm_tracked': np.random.choice([True, False], p=[0.6, 0.4]),
            'click_through_rate': round(np.random.uniform(0.5, 5.5) * utm_multiplier, 2),
            'social_promoted': np.random.choice([True, False], p=[0.55, 0.45]),
            'sentiment_tone': np.random.choice(['positive', 'neutral', 'mixed'], p=[0.65, 0.25, 0.10]),
        })
    
    return pd.DataFrame(data)

def generate_timeline_data():
    """Generate day-by-day coverage timeline."""
    days = [0, 1, 2, 3, 4, 5, 7, 10, 14, 21, 30]
    
    data = []
    cumulative_pickup = 0
    cumulative_impressions = 0
    
    for day in days:
        if day == 0:
            cumulative_pickup = 1
            cumulative_impressions = 0
        else:
            new_pickups = np.random.randint(1, 5) if day <= 5 else np.random.randint(0, 3)
            new_impressions = np.random.randint(1200, 6000) * (1 if day <= 7 else 0.7)
            cumulative_pickup += new_pickups
            cumulative_impressions += int(new_impressions)
        
        data.append({
            'day': day,
            'cumulative_pickups': cumulative_pickup,
            'daily_impressions': int(new_impressions) if day > 0 else 0,
            'cumulative_impressions': cumulative_impressions,
        })
    
    return pd.DataFrame(data)

# ─── Step 2: Analysis Functions ───
def analyze_outlet_performance(df):
    """Calculate outlet-level metrics."""
    summary = df.groupby(['tier', 'region']).agg(
        avg_pickup=('pickup_rate', 'mean'),
        avg_impressions=('impressions', 'mean'),
        total_outlets=('outlet', 'count'),
        avg_ctr=('click_through_rate', 'mean'),
    ).reset_index()
    return summary

def analyze_geographic_distribution(df):
    """Calculate regional distribution."""
    geo = df.groupby('region').agg(
        total_impressions=('impressions', 'sum'),
        avg_pickup=('pickup_rate', 'mean'),
        outlet_count=('outlet', 'count'),
    ).reset_index()
    geo['share'] = (geo['total_impressions'] / geo['total_impressions'].sum() * 100).round(1)
    return geo

# ─── Step 3: Visualizations ───
def create_outlet_bar_chart(df):
    """Create horizontal bar chart of pickup rates by outlet."""
    df_sorted = df.sort_values('pickup_rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = [NAVY if row['tier'] == 1 else AMBER for _, row in df_sorted.iterrows()]
    
    bars = ax.barh(df_sorted['outlet'], df_sorted['pickup_rate'], color=colors, height=0.7)
    
    for bar, val in zip(bars, df_sorted['pickup_rate']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Media Pickup Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Press Release Pickup Rate by Media Outlet', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlim(0, 110)
    
    # Add tier legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=NAVY, label='Tier 1 Outlets'),
                       Patch(facecolor=AMBER, label='Tier 2 Outlets')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('output/outlet_pickup_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Outlet pickup rate chart saved.")

def create_impressions_timeline(df_timeline):
    """Create area chart of cumulative impressions over time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.fill_between(df_timeline['day'], df_timeline['cumulative_impressions'],
                    alpha=0.2, color=TEAL)
    ax.plot(df_timeline['day'], df_timeline['cumulative_impressions'],
            'o-', color=TEAL, linewidth=2.5, markersize=7, label='Cumulative Impressions')
    
    # Add daily bars
    ax.bar(df_timeline['day'], df_timeline['daily_impressions'],
           alpha=0.3, color=NAVY, width=1.5, label='Daily Impressions')
    
    ax.set_xlabel('Days Post-Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Impressions', fontsize=12, fontweight='bold')
    ax.set_title('Press Release Coverage Timeline', fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Day {int(x)}' if x > 0 else 'Day 0'))
    
    plt.tight_layout()
    plt.savefig('output/coverage_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Coverage timeline chart saved.")

def create_geographic_chart(df_geo):
    """Create geographic distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_geo = df_geo.sort_values('share', ascending=True)
    bars = ax.barh(df_geo['region'], df_geo['share'], color=AMBER, height=0.6)
    
    for bar, val in zip(bars, df_geo['share']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Share of Total Impressions (%)', fontsize=12, fontweight='bold')
    ax.set_title('Geographic Distribution of Press Release Coverage', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlim(0, 55)
    
    plt.tight_layout()
    plt.savefig('output/geographic_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Geographic distribution chart saved.")

def create_roi_scatter(df):
    """Create scatter plot of impressions vs. CTR by tier."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    tier_colors = {1: NAVY, 2: AMBER}
    
    for tier in df['tier'].unique():
        subset = df[df['tier'] == tier]
        ax.scatter(subset['impressions'], subset['click_through_rate'],
                  s=120, alpha=0.75, color=tier_colors[tier],
                  label=f'Tier {tier}', edgecolors='white', linewidth=0.8)
        # Add outlet labels
        for _, row in subset.iterrows():
            ax.annotate(row['outlet'], (row['impressions'], row['click_through_rate']),
                       textcoords="offset points", xytext=(8, 5), fontsize=9, color=NAVY)
    
    ax.set_xlabel('Audience Impressions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Click-Through Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('ROI Correlation: Impressions vs. Click-Through Rate', fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('output/roi_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ ROI scatter plot saved.")

# ─── Step 4: Executive Report ───
def generate_report(df, df_timeline, df_geo, outlet_summary):
    """Generate text-based executive report."""
    total_impressions = df['impressions'].sum()
    avg_pickup = df['pickup_rate'].mean()
    tier1_pickup = df[df['tier'] == 1]['pickup_rate'].mean()
    tier2_pickup = df[df['tier'] == 2]['pickup_rate'].mean()
    avg_ctr = df['click_through_rate'].mean()
    utm_ctr = df[df['utm_tracked'] == True]['click_through_rate'].mean()
    organic_ctr = df[df['utm_tracked'] == False]['click_through_rate'].mean()
    
    report = []
    report.append("=" * 70)
    report.append("PRESS RELEASE PERFORMANCE — EXECUTIVE SUMMARY")
    report.append("=" * 70)
    report.append(f"\nTotal Media Outlets Targeted: {len(df)}")
    report.append(f"Total Impressions: {total_impressions:,}")
    report.append(f"Average Pickup Rate: {avg_pickup:.1f}%")
    report.append(f"Average Click-Through Rate: {avg_ctr:.2f}%")
    report.append(f"\n{'─' * 50}")
    report.append("TIER COMPARISON")
    report.append(f"{'─' * 50}")
    report.append(f"  Tier 1 Average Pickup:  {tier1_pickup:.1f}%")
    report.append(f"  Tier 2 Average Pickup:  {tier2_pickup:.1f}%")
    report.append(f"  Lift (Tier 1 vs Tier 2): {((tier1_pickup - tier2_pickup) / tier2_pickup * 100):.1f}%")
    report.append(f"\n{'─' * 50}")
    report.append("UTM TRACKING ANALYSIS")
    report.append(f"{'─' * 50}")
    report.append(f"  UTM-Tracked CTR:  {utm_ctr:.2f}%")
    report.append(f"  Organic CTR:      {organic_ctr:.2f}%")
    report.append(f"  UTM Lift:         {((utm_ctr - organic_ctr) / organic_ctr * 100):.1f}x")
    report.append(f"\n{'─' * 50}")
    report.append("GEOGRAPHIC DISTRIBUTION")
    report.append(f"{'─' * 50}")
    for _, row in df_geo.iterrows():
        report.append(f"  {row['region']:20s}: {row['share']:5.1f}% (Impressions: {row['total_impressions']:,})")
    report.append(f"\n{'─' * 50}")
    report.append("PEAK COVERAGE")
    report.append(f"{'─' * 50}")
    peak_day = df_timeline.loc[df_timeline['daily_impressions'].idxmax()]
    report.append(f"  Peak Day: Day {int(peak_day['day'])}")
    report.append(f"  Peak Daily Impressions: {peak_day['daily_impressions']:,}")
    report.append(f"  30-Day Cumulative: {df_timeline['cumulative_impressions'].max():,}")
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    with open('output/press_release_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n✓ Report saved to output/press_release_report.txt")

# ─── Main Execution ───
if __name__ == '__main__':
    print("\n📊 Press Release Performance Analytics Pipeline")
    print("=" * 50)
    
    print("\n[1/4] Generating media outlet dataset...")
    df = generate_outlet_data()
    df_timeline = generate_timeline_data()
    print(f"      {len(df)} outlets tracked across {df['region'].nunique()} regions")
    
    print("\n[2/4] Running analysis...")
    outlet_summary = analyze_outlet_performance(df)
    df_geo = analyze_geographic_distribution(df)
    
    print("\n[3/4] Generating visualizations...")
    create_outlet_bar_chart(df)
    create_impressions_timeline(df_timeline)
    create_geographic_chart(df_geo)
    create_roi_scatter(df)
    
    print("\n[4/4] Generating executive report...")
    generate_report(df, df_timeline, df_geo, outlet_summary)
    
    print("\n✅ Analysis complete. All outputs saved to ./output/")
