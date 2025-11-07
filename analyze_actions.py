#!/usr/bin/env python3
"""
Explainability Dashboard for SustainDC MARL Policies
Analyzes action logs to understand what the agents learned.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class PolicyAnalyzer:
    """Analyze and visualize MARL policy actions."""

    def __init__(self, csv_path):
        """Load action logs from CSV."""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(csv_path).parent / 'analysis'
        self.output_dir.mkdir(exist_ok=True)
        print(f"Loaded {len(self.df)} timesteps from {csv_path}")
        print(f"Columns: {self.df.columns.tolist()}")

    def action_distribution(self):
        """Plot action distribution for each agent."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # LS Agent actions
        ls_actions = {0: 'Defer', 1: 'Hold', 2: 'Execute'}
        ls_counts = self.df['ls_action'].value_counts()
        ls_pcts = (ls_counts / len(self.df) * 100).to_dict()

        axes[0].bar([ls_actions.get(k, k) for k in ls_counts.index], ls_counts.values, color='skyblue')
        axes[0].set_title('Load Shifting Agent Actions', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, (action, count) in enumerate(ls_counts.items()):
            axes[0].text(i, count, f'{ls_pcts[action]:.1f}%', ha='center', va='bottom')

        # DC Agent actions (assuming -1, 0, +1)
        dc_actions = {-1: 'Lower', 0: 'Hold', 1: 'Raise'}
        dc_counts = self.df['dc_crac_setpoint_delta'].value_counts()
        dc_pcts = (dc_counts / len(self.df) * 100).to_dict()

        axes[1].bar([dc_actions.get(k, k) for k in dc_counts.index], dc_counts.values, color='lightcoral')
        axes[1].set_title('HVAC Agent Actions (Setpoint)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count')
        for i, (action, count) in enumerate(dc_counts.items()):
            axes[1].text(i, count, f'{dc_pcts[action]:.1f}%', ha='center', va='bottom')

        # Battery Agent actions
        if 'bat_action' in self.df.columns:
            bat_actions = {0: 'Charge', 1: 'Idle', 2: 'Discharge'}
            bat_counts = self.df['bat_action'].value_counts()
            bat_pcts = (bat_counts / len(self.df) * 100).to_dict()

            axes[2].bar([bat_actions.get(k, k) for k in bat_counts.index], bat_counts.values, color='lightgreen')
            axes[2].set_title('Battery Agent Actions', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Count')
            for i, (action, count) in enumerate(bat_counts.items()):
                axes[2].text(i, count, f'{bat_pcts[action]:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        save_path = self.output_dir / 'action_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved action distribution to {save_path}")
        plt.close()

    def actions_over_time(self):
        """Plot actions over time as heatmap."""
        # Prepare data - sample every 4 timesteps for readability
        sample_interval = 4
        df_sample = self.df.iloc[::sample_interval].copy()
        df_sample['timestep'] = range(len(df_sample))

        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

        # LS Actions
        ls_pivot = df_sample.pivot_table(index='ls_action', columns='timestep', aggfunc='size', fill_value=0)
        sns.heatmap([[row] for row in df_sample['ls_action']], ax=axes[0], cmap='RdYlGn_r',
                    cbar_kws={'label': 'Action'}, yticklabels=False, xticklabels=False)
        axes[0].set_title('Load Shifting Actions Over Time (Red=Defer, Yellow=Hold, Green=Execute)')
        axes[0].set_ylabel('LS Agent')

        # DC Actions
        sns.heatmap([[row] for row in df_sample['dc_crac_setpoint_delta']], ax=axes[1], cmap='coolwarm',
                    cbar_kws={'label': 'Setpoint Δ'}, yticklabels=False, xticklabels=False)
        axes[1].set_title('HVAC Setpoint Changes (Blue=Lower, White=Hold, Red=Raise)')
        axes[1].set_ylabel('DC Agent')

        # Battery Actions
        if 'bat_action' in df_sample.columns:
            sns.heatmap([[row] for row in df_sample['bat_action']], ax=axes[2], cmap='PRGn',
                        cbar_kws={'label': 'Action'}, yticklabels=False)
            axes[2].set_title('Battery Actions (Purple=Charge, White=Idle, Green=Discharge)')
            axes[2].set_ylabel('BAT Agent')
            axes[2].set_xlabel(f'Time (15-min intervals, sampled every {sample_interval} steps)')

        plt.tight_layout()
        save_path = self.output_dir / 'actions_timeline.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved actions timeline to {save_path}")
        plt.close()

    def action_environment_correlation(self):
        """Plot how actions correlate with environmental conditions."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('LS Action vs Carbon Intensity', 'LS Action vs Queue Size',
                           'DC Action vs Ambient Temp', 'Battery Action vs CI'),
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "box"}]]
        )

        # LS vs CI
        if 'bat_avg_CI' in self.df.columns:
            for action in sorted(self.df['ls_action'].unique()):
                data = self.df[self.df['ls_action'] == action]['bat_avg_CI']
                fig.add_trace(go.Box(y=data, name=f'Action {int(action)}',
                                     marker_color='skyblue'), row=1, col=1)

        # LS vs Queue
        for action in sorted(self.df['ls_action'].unique()):
            data = self.df[self.df['ls_action'] == action]['ls_tasks_in_queue']
            fig.add_trace(go.Box(y=data, name=f'Action {int(action)}',
                                 marker_color='lightgreen'), row=1, col=2)

        # DC vs Ambient Temp
        if 'outside_temp' in self.df.columns:
            for action in sorted(self.df['dc_crac_setpoint_delta'].unique()):
                data = self.df[self.df['dc_crac_setpoint_delta'] == action]['outside_temp']
                fig.add_trace(go.Box(y=data, name=f'Δ={int(action)}',
                                     marker_color='coral'), row=2, col=1)

        # Battery vs CI
        if 'bat_action' in self.df.columns and 'bat_avg_CI' in self.df.columns:
            for action in sorted(self.df['bat_action'].unique()):
                data = self.df[self.df['bat_action'] == action]['bat_avg_CI']
                fig.add_trace(go.Box(y=data, name=f'Action {int(action)}',
                                     marker_color='gold'), row=2, col=2)

        fig.update_layout(height=800, showlegend=False, title_text="Action-Environment Correlations")
        fig.update_yaxes(title_text="CI (kg/kWh)", row=1, col=1)
        fig.update_yaxes(title_text="Queue Size", row=1, col=2)
        fig.update_yaxes(title_text="Temp (°C)", row=2, col=1)
        fig.update_yaxes(title_text="CI (kg/kWh)", row=2, col=2)

        save_path = self.output_dir / 'action_correlations.html'
        fig.write_html(str(save_path))
        print(f"✓ Saved action correlations to {save_path}")

    def power_co2_analysis(self):
        """Analyze power and CO2 impact of actions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Power consumption by LS action
        ls_group = self.df.groupby('ls_action')['dc_total_power_kW'].mean()
        ls_group.plot(kind='bar', ax=axes[0,0], color='steelblue')
        axes[0,0].set_title('Avg Total Power by LS Action')
        axes[0,0].set_ylabel('Power (kW)')
        ls_labels = {0: 'Defer', 1: 'Hold', 2: 'Execute'}
        axes[0,0].set_xticklabels([ls_labels.get(int(x), x) for x in ls_group.index], rotation=0)

        # HVAC power by DC action
        dc_group = self.df.groupby('dc_crac_setpoint_delta')['dc_HVAC_total_power_kW'].mean()
        dc_group.plot(kind='bar', ax=axes[0,1], color='coral')
        axes[0,1].set_title('Avg HVAC Power by DC Action')
        axes[0,1].set_ylabel('HVAC Power (kW)')
        dc_labels = {-1: 'Lower', 0: 'Hold', 1: 'Raise'}
        axes[0,1].set_xticklabels([dc_labels.get(int(x), x) for x in dc_group.index], rotation=0)

        # CO2 by Battery action
        if 'bat_CO2_footprint' in self.df.columns and 'bat_action' in self.df.columns:
            bat_group = self.df.groupby('bat_action')['bat_CO2_footprint'].mean()
            bat_group.plot(kind='bar', ax=axes[1,0], color='lightgreen')
            axes[1,0].set_title('Avg CO2 Footprint by Battery Action')
            axes[1,0].set_ylabel('CO2 (kg)')
            bat_labels = {0: 'Charge', 1: 'Idle', 2: 'Discharge'}
            axes[1,0].set_xticklabels([bat_labels.get(int(x), x) for x in bat_group.index], rotation=0)

        # Power over time
        axes[1,1].plot(self.df['dc_total_power_kW'], label='Total Power', alpha=0.7)
        axes[1,1].plot(self.df['dc_ITE_total_power_kW'], label='IT Power', alpha=0.7)
        axes[1,1].plot(self.df['dc_HVAC_total_power_kW'], label='HVAC Power', alpha=0.7)
        axes[1,1].set_title('Power Consumption Over Time')
        axes[1,1].set_xlabel('Timestep')
        axes[1,1].set_ylabel('Power (kW)')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'power_co2_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved power/CO2 analysis to {save_path}")
        plt.close()

    def extract_policy_rules(self):
        """Extract interpretable decision rules."""
        rules = []
        rules.append("=" * 60)
        rules.append("LEARNED POLICY RULES (Interpretable Insights)")
        rules.append("=" * 60)

        # LS Agent rules
        rules.append("\n🔵 LOAD SHIFTING AGENT:")
        if 'bat_avg_CI' in self.df.columns:
            high_ci = self.df['bat_avg_CI'].quantile(0.75)
            low_ci = self.df['bat_avg_CI'].quantile(0.25)

            defer_rate_high_ci = (self.df[self.df['bat_avg_CI'] > high_ci]['ls_action'] == 0).mean() * 100
            defer_rate_low_ci = (self.df[self.df['bat_avg_CI'] < low_ci]['ls_action'] == 0).mean() * 100

            rules.append(f"  • When CI > {high_ci:.3f} kg/kWh → Defers tasks {defer_rate_high_ci:.1f}% of time")
            rules.append(f"  • When CI < {low_ci:.3f} kg/kWh → Defers tasks {defer_rate_low_ci:.1f}% of time")

        avg_queue = self.df['ls_tasks_in_queue'].mean()
        rules.append(f"  • Average task queue size: {avg_queue:.0f} tasks")
        rules.append(f"  • Tasks dropped: {self.df['ls_tasks_dropped'].sum():.0f} total")

        # DC Agent rules
        rules.append("\n🔴 HVAC AGENT:")
        if 'outside_temp' in self.df.columns:
            hot_temp = self.df['outside_temp'].quantile(0.75)
            cold_temp = self.df['outside_temp'].quantile(0.25)

            lower_rate_hot = (self.df[self.df['outside_temp'] > hot_temp]['dc_crac_setpoint_delta'] == -1).mean() * 100
            raise_rate_cold = (self.df[self.df['outside_temp'] < cold_temp]['dc_crac_setpoint_delta'] == 1).mean() * 100

            rules.append(f"  • When ambient > {hot_temp:.1f}°C → Lowers setpoint {lower_rate_hot:.1f}% of time")
            rules.append(f"  • When ambient < {cold_temp:.1f}°C → Raises setpoint {raise_rate_cold:.1f}% of time")

        avg_hvac = self.df['dc_HVAC_total_power_kW'].mean()
        avg_it = self.df['dc_ITE_total_power_kW'].mean()
        pue = (avg_hvac + avg_it) / avg_it
        rules.append(f"  • Average HVAC power: {avg_hvac:.1f} kW")
        rules.append(f"  • Average IT power: {avg_it:.1f} kW")
        rules.append(f"  • Effective PUE: {pue:.3f}")

        # Battery Agent rules
        if 'bat_action' in self.df.columns and 'bat_avg_CI' in self.df.columns:
            rules.append("\n🟢 BATTERY AGENT:")
            charge_rate_low_ci = (self.df[self.df['bat_avg_CI'] < low_ci]['bat_action'] == 0).mean() * 100
            discharge_rate_high_ci = (self.df[self.df['bat_avg_CI'] > high_ci]['bat_action'] == 2).mean() * 100

            rules.append(f"  • When CI < {low_ci:.3f} kg/kWh → Charges {charge_rate_low_ci:.1f}% of time")
            rules.append(f"  • When CI > {high_ci:.3f} kg/kWh → Discharges {discharge_rate_high_ci:.1f}% of time")

            if 'bat_SOC' in self.df.columns:
                avg_soc = self.df['bat_SOC'].mean()
                rules.append(f"  • Average battery SOC: {avg_soc:.1f}%")

        # Overall metrics
        rules.append("\n📊 OVERALL PERFORMANCE:")
        total_energy = self.df['dc_total_power_kW'].sum() * 0.25 / 1000  # kWh (15min intervals)
        if 'bat_CO2_footprint' in self.df.columns:
            avg_co2 = self.df['bat_CO2_footprint'].mean()
            rules.append(f"  • Total energy consumed: {total_energy:.1f} MWh")
            rules.append(f"  • Average CO2 per timestep: {avg_co2:.1f} kg")

        rules.append("=" * 60)

        # Save rules
        rules_text = "\n".join(rules)
        save_path = self.output_dir / 'policy_rules.txt'
        with open(save_path, 'w') as f:
            f.write(rules_text)
        print(f"\n{rules_text}\n")
        print(f"✓ Saved policy rules to {save_path}")

    def generate_dashboard(self):
        """Generate complete explainability dashboard."""
        print("\n" + "="*60)
        print("GENERATING EXPLAINABILITY DASHBOARD")
        print("="*60)

        self.action_distribution()
        self.actions_over_time()
        self.action_environment_correlation()
        self.power_co2_analysis()
        self.extract_policy_rules()

        print("\n" + "="*60)
        print(f"✅ DASHBOARD COMPLETE! All files saved to: {self.output_dir}")
        print("="*60)
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  📁 {file.name}")

def main():
    """Main entry point."""
    # Find most recent evaluation CSV
    import glob
    csv_files = glob.glob('results/sustaindc/ny/happo/ny_quicktest/*/evaluation_data/all_agents_episode_*.csv')

    if not csv_files:
        print("❌ No evaluation CSV files found!")
        print("Run eval_sustaindc.py first to generate action logs.")
        return

    # Use most recent
    csv_path = max(csv_files, key=os.path.getmtime)
    print(f"\n📂 Analyzing: {csv_path}\n")

    analyzer = PolicyAnalyzer(csv_path)
    analyzer.generate_dashboard()

if __name__ == "__main__":
    main()
