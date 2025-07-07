#!/usr/bin/env python3
"""
Week 3 Speed Estimation - Calibration Statistics Viewer
=======================================================

This script displays current calibration statistics without running the robot.
Shows individual run data and cumulative calibration parameters.

Usage:
    python show_stats.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import sys

class CalibrationStatsViewer:
    def __init__(self):
        """Initialize calibration statistics viewer"""
        self.csv_file = Path("calibration_data.csv")
        
        if not self.csv_file.exists():
            print("❌ No calibration data found!")
            print("   Run calibration_script.py first to collect data")
            sys.exit(1)
        
        # Load data
        self.df = pd.read_csv(self.csv_file)
        print(f"📖 Loaded calibration data from: {self.csv_file}")

    def show_summary_stats(self):
        """Display summary statistics"""
        total_runs = len(self.df)
        calibration_runs = len(self.df[self.df['include_in_calibration'] == True])
        
        print(f"\n{'='*60}")
        print(f"📊 CALIBRATION STATISTICS SUMMARY")
        print(f"{'='*60}")
        print(f"Total runs: {total_runs}")
        print(f"Runs used for calibration: {calibration_runs}")
        
        if total_runs > 0:
            last_updated = self.df['timestamp'].iloc[-1]
            print(f"Last updated: {last_updated}")

    def show_current_calibration(self):
        """Display current calibration parameters"""
        calibration_data = self.df[self.df['include_in_calibration'] == True]
        
        if len(calibration_data) < 2:
            print(f"\n⚠️  INSUFFICIENT DATA FOR CALIBRATION")
            print(f"   Need at least 2 runs, currently have {len(calibration_data)}")
            return
        
        # Get latest cumulative stats
        latest_row = self.df.iloc[-1]
        slope = latest_row['cumulative_slope']
        intercept = latest_row['cumulative_intercept']
        r_squared = latest_row['cumulative_r_squared']
        n_points = int(latest_row['cumulative_n_points'])
        
        print(f"\n🎯 CURRENT CALIBRATION:")
        print(f"   speed = {slope:.6f} × flow + ({intercept:.6f})")
        print(f"   R-squared: {r_squared:.3f}")
        print(f"   Based on {n_points} data points")
        
        # Quality assessment
        if r_squared >= 0.95:
            quality = "Excellent! 🌟"
        elif r_squared >= 0.90:
            quality = "Very Good ✅"
        elif r_squared >= 0.80:
            quality = "Good 👍"
        elif r_squared >= 0.70:
            quality = "Fair ⚠️"
        else:
            quality = "Poor - collect more data ❌"
        
        print(f"   Calibration quality: {quality}")
        
        print(f"\n📋 COPY TO SPEED_ESTIMATOR.PY:")
        print(f"   self.flow_to_speed_slope = {slope:.6f}")
        print(f"   self.flow_to_speed_intercept = {intercept:.6f}")

    def show_individual_runs(self):
        """Display individual run data"""
        print(f"\n📝 INDIVIDUAL RUN DATA:")
        print(f"{'Run':<4} {'Distance':<8} {'Power':<6} {'Speed':<8} {'Flow':<8} {'Features':<9} {'Include'}")
        print(f"{'-'*60}")
        
        for _, row in self.df.iterrows():
            run_id = int(row['run_id'])
            distance = row['distance']
            power = int(row['motor_power'])
            speed = row['calculated_speed']
            flow = row['avg_optical_flow']
            features = int(row['num_features'])
            include = "✓" if row['include_in_calibration'] else "✗"
            
            print(f"{run_id:<4} {distance:<8.1f} {power:<6}% {speed:<8.3f} {flow:<8.1f} {features:<9} {include}")

    def show_data_distribution(self):
        """Show data distribution statistics"""
        calibration_data = self.df[self.df['include_in_calibration'] == True]
        
        if len(calibration_data) == 0:
            return
        
        print(f"\n📈 DATA DISTRIBUTION:")
        
        # Speed range
        speed_min = calibration_data['calculated_speed'].min()
        speed_max = calibration_data['calculated_speed'].max()
        speed_mean = calibration_data['calculated_speed'].mean()
        
        print(f"   Speed range: {speed_min:.3f} - {speed_max:.3f} m/s (avg: {speed_mean:.3f})")
        
        # Flow range
        flow_min = calibration_data['avg_optical_flow'].min()
        flow_max = calibration_data['avg_optical_flow'].max()
        flow_mean = calibration_data['avg_optical_flow'].mean()
        
        print(f"   Flow range: {flow_min:.1f} - {flow_max:.1f} px/frame (avg: {flow_mean:.1f})")
        
        # Distance and power coverage
        distances = calibration_data['distance'].unique()
        powers = calibration_data['motor_power'].unique()
        
        print(f"   Distances tested: {sorted(distances)} meters")
        print(f"   Power levels tested: {sorted(powers)}%")

    def create_scatter_plot(self, save_plot=False):
        """Create scatter plot of flow vs speed"""
        calibration_data = self.df[self.df['include_in_calibration'] == True]
        
        if len(calibration_data) < 2:
            print("⚠️  Not enough data for scatter plot")
            return
        
        flows = calibration_data['avg_optical_flow'].values
        speeds = calibration_data['calculated_speed'].values
        
        # Get calibration line
        latest_row = self.df.iloc[-1]
        slope = latest_row['cumulative_slope']
        intercept = latest_row['cumulative_intercept']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(flows, speeds, c='blue', alpha=0.7, s=100, edgecolors='black')
        
        # Plot calibration line
        flow_range = np.linspace(flows.min() * 0.9, flows.max() * 1.1, 100)
        speed_line = slope * flow_range + intercept
        plt.plot(flow_range, speed_line, 'r--', linewidth=2, 
                label=f'Calibration: speed = {slope:.4f} × flow + {intercept:.4f}')
        
        plt.xlabel('Optical Flow (pixels/frame)')
        plt.ylabel('Speed (m/s)')
        plt.title('Optical Flow vs Speed Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r_squared = latest_row['cumulative_r_squared']
        plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_plot:
            plot_path = Path("calibration_plot.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()

    def show_recommendations(self):
        """Show recommendations for improving calibration"""
        calibration_data = self.df[self.df['include_in_calibration'] == True]
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if len(calibration_data) < 3:
            print("   • Collect more data points (aim for 5-8 runs)")
        
        if len(calibration_data) > 0:
            # Check speed range
            speed_range = calibration_data['calculated_speed'].max() - calibration_data['calculated_speed'].min()
            if speed_range < 0.3:
                print("   • Try more varied motor power settings for wider speed range")
            
            # Check R-squared
            if len(calibration_data) >= 2:
                r_squared = self.df.iloc[-1]['cumulative_r_squared']
                if r_squared < 0.90:
                    print("   • Consider collecting more data to improve fit quality")
                    print("   • Check videos for consistent feature tracking")
        
        # Check distance variety
        distances = calibration_data['distance'].unique() if len(calibration_data) > 0 else []
        if len(distances) < 2:
            print("   • Test multiple distance settings (2.0m, 2.5m, 3.0m)")
        
        # Check power variety
        powers = calibration_data['motor_power'].unique() if len(calibration_data) > 0 else []
        if len(powers) < 2:
            print("   • Test multiple power settings (30%, 40%, 50%)")

    def toggle_run_inclusion(self, run_id):
        """Toggle whether a specific run is included in calibration"""
        if run_id < 1 or run_id > len(self.df):
            print(f"❌ Invalid run ID: {run_id}")
            return
        
        # Toggle inclusion status
        current_status = self.df.loc[self.df['run_id'] == run_id, 'include_in_calibration'].iloc[0]
        new_status = not current_status
        self.df.loc[self.df['run_id'] == run_id, 'include_in_calibration'] = new_status
        
        # Recalculate cumulative statistics
        self._recalculate_cumulative_stats()
        
        # Save updated CSV
        self.df.to_csv(self.csv_file, index=False)
        
        status_text = "included" if new_status else "excluded"
        print(f"✅ Run {run_id} {status_text} from calibration")

    def _recalculate_cumulative_stats(self):
        """Recalculate cumulative statistics after toggling run inclusion"""
        calibration_data = self.df[self.df['include_in_calibration'] == True]
        
        if len(calibration_data) >= 2:
            flows = calibration_data['avg_optical_flow'].values
            speeds = calibration_data['calculated_speed'].values
            
            # Linear regression
            slope, intercept = np.polyfit(flows, speeds, 1)
            
            # Calculate R-squared
            predicted_speeds = slope * flows + intercept
            ss_res = np.sum((speeds - predicted_speeds) ** 2)
            ss_tot = np.sum((speeds - np.mean(speeds)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            n_points = len(calibration_data)
        else:
            slope, intercept, r_squared, n_points = 0, 0, 0, len(calibration_data)
        
        # Update all rows with new cumulative stats
        self.df['cumulative_slope'] = slope
        self.df['cumulative_intercept'] = intercept
        self.df['cumulative_r_squared'] = r_squared
        self.df['cumulative_n_points'] = n_points

    def interactive_menu(self):
        """Display interactive menu for data management"""
        while True:
            print(f"\n{'='*40}")
            print("📋 CALIBRATION DATA MANAGER")
            print(f"{'='*40}")
            print("1. Show summary")
            print("2. Show current calibration")
            print("3. Show individual runs")
            print("4. Show data distribution")
            print("5. Create scatter plot")
            print("6. Toggle run inclusion")
            print("7. Show recommendations")
            print("8. Exit")
            
            try:
                choice = input("\nSelect option (1-8): ").strip()
                
                if choice == '1':
                    self.show_summary_stats()
                elif choice == '2':
                    self.show_current_calibration()
                elif choice == '3':
                    self.show_individual_runs()
                elif choice == '4':
                    self.show_data_distribution()
                elif choice == '5':
                    try:
                        import matplotlib.pyplot as plt
                        self.create_scatter_plot(save_plot=True)
                    except ImportError:
                        print("⚠️  Matplotlib not available for plotting")
                elif choice == '6':
                    self.show_individual_runs()
                    try:
                        run_id = int(input("Enter run ID to toggle: "))
                        self.toggle_run_inclusion(run_id)
                    except ValueError:
                        print("❌ Please enter a valid run ID number")
                elif choice == '7':
                    self.show_recommendations()
                elif choice == '8':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-8.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function - show all stats by default or interactive mode"""
    import sys
    
    viewer = CalibrationStatsViewer()
    
    # If run with --interactive flag, show menu
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        viewer.interactive_menu()
    else:
        # Default: show all statistics
        viewer.show_summary_stats()
        viewer.show_current_calibration()
        viewer.show_individual_runs()
        viewer.show_data_distribution()
        viewer.show_recommendations()


if __name__ == "__main__":
    main()