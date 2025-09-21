#!/usr/bin/env python3
"""
Figure and Table Generation Script for Cite-and-Challenge Peer Protocol
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger
import sys

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"

# Configure logger
logger.add("logs/figure_generation.log", rotation="30 MB")
logger.add(
    sys.stdout,
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level: <7}}|{CYAN}{{name: >12.12}}{END}.{CYAN}{{function: <22.22}}{END}:{CYAN}{{line: <4}}{END}| {{message}}",
    level="INFO",
    colorize=False
)

class FigureGenerator:
    """Generates publication-quality figures and tables for experiment results."""
    
    def __init__(self, results_file: str, output_dir: str = "figures"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load experiment results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Set up matplotlib style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        logger.info(f"{BLUE}Figure generator initialized{END}")

    def create_performance_comparison_chart(self) -> str:
        """Creates a performance comparison chart showing system vs baseline metrics."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        accuracy_data = self.results['detailed_results']['comparison_analysis_summary']['accuracy']
        system_acc = float(accuracy_data['system'])
        baseline_acc = float(accuracy_data['baseline'])
        
        categories = ['System', 'Baseline']
        accuracies = [system_acc, baseline_acc]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars1 = ax1.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Quality comparison
        quality_data = self.results['detailed_results']['comparison_analysis_summary']['quality']
        system_qual = float(quality_data['system'])
        baseline_qual = float(quality_data['baseline'])
        
        qualities = [system_qual, baseline_qual]
        
        bars2 = ax2.bar(categories, qualities, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Quality Comparison', fontweight='bold')
        ax2.set_ylabel('Quality Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, qual in zip(bars2, qualities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{qual:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Cite-and-Challenge System Performance vs Baseline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'performance_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"{GREEN}Performance comparison chart saved to {output_file}{END}")
        return str(output_file)

    def create_metrics_radar_chart(self) -> str:
        """Creates a radar chart showing all key metrics."""
        
        # Prepare data
        metrics_data = self.results['key_metrics']
        
        # Normalize metrics to 0-1 scale for visualization
        metrics = {
            'Accuracy': metrics_data['overall_accuracy'],
            'Citation Quality': metrics_data['citation_quality'],
            'Evidence Strength': metrics_data['evidence_strength'],
            'Processing Efficiency': min(metrics_data['processing_efficiency'] / 5.0, 1.0),  # Normalize
            'Challenge Effectiveness': metrics_data['challenge_effectiveness'],
            'Revision Success': metrics_data['revision_success_rate']
        }
        
        # Number of variables
        categories = list(metrics.keys())
        N = len(categories)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Add the values for our system
        values = list(metrics.values())
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Cite-and-Challenge System', color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title('System Performance Metrics Overview', size=16, fontweight='bold', pad=20)
        
        output_file = self.output_dir / 'metrics_radar.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"{GREEN}Metrics radar chart saved to {output_file}{END}")
        return str(output_file)

    def create_improvement_analysis_chart(self) -> str:
        """Creates a chart showing improvement areas and their significance."""
        
        improvement_data = self.results['improvement_analysis']['vs_random_baseline']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Data for the chart
        metrics = ['Accuracy', 'Quality']
        improvements = [
            improvement_data['accuracy_improvement'],
            improvement_data['quality_improvement']
        ]
        
        # Create colors based on improvement (red for negative, green for positive)
        colors = ['red' if imp < 0 else 'green' for imp in improvements]
        
        bars = ax.bar(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Customize the plot
        ax.set_title('Performance Improvement vs Random Baseline', fontsize=16, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=14)
        ax.set_xlabel('Metrics', fontsize=14)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (5 if height > 0 else -10),
                   f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                   fontweight='bold', fontsize=12)
        
        # Add significance indicators
        significance_text = "Statistical Significance: " + str(improvement_data['statistically_significant'])
        ax.text(0.02, 0.98, significance_text, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'improvement_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"{GREEN}Improvement analysis chart saved to {output_file}{END}")
        return str(output_file)

    def create_detailed_metrics_table(self) -> str:
        """Creates a detailed metrics table in LaTeX format."""
        
        # Extract detailed performance metrics
        system_perf = self.results['detailed_results']['system_performance']
        comparison = self.results['detailed_results']['comparison_analysis_summary']
        
        # Create DataFrame for the table
        metrics_data = [
            ['Overall Accuracy', f"{system_perf['overall_accuracy']:.3f}", f"{comparison['accuracy']['baseline']}", f"{comparison['accuracy']['improvement']}", comparison['accuracy']['significant']],
            ['Citation Accuracy', f"{system_perf['citation_accuracy']:.3f}", "N/A", "N/A", "N/A"],
            ['Evidence Accuracy', f"{system_perf['evidence_accuracy']:.3f}", "N/A", "N/A", "N/A"],
            ['Response Quality', f"{system_perf['avg_response_quality']:.3f}", f"{comparison['quality']['baseline']}", f"{comparison['quality']['improvement']}", comparison['quality']['significant']],
            ['Citation Quality', f"{system_perf['avg_citation_quality']:.3f}", "N/A", "N/A", "N/A"],
            ['Evidence Strength', f"{system_perf['avg_evidence_strength']:.3f}", "N/A", "N/A", "N/A"],
            ['Challenge Precision', f"{system_perf['challenge_precision']:.3f}", "N/A", "N/A", "N/A"],
            ['Challenge Recall', f"{system_perf['challenge_recall']:.3f}", "N/A", "N/A", "N/A"],
            ['Challenge F1-Score', f"{system_perf['challenge_f1']:.3f}", "N/A", "N/A", "N/A"],
            ['Processing Time (s)', f"{system_perf['avg_processing_time']:.3f}", "N/A", "N/A", "N/A"],
            ['Token Efficiency', f"{system_perf['token_efficiency']:.6f}", "N/A", "N/A", "N/A"],
            ['Throughput (claims/min)', f"{system_perf['throughput_per_minute']:.2f}", "N/A", "N/A", "N/A"]
        ]
        
        # Create LaTeX table
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Detailed Performance Metrics of the Cite-and-Challenge System}
\\label{tab:detailed_metrics}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{System Score} & \\textbf{Baseline} & \\textbf{Improvement (\\%)} & \\textbf{Significant} \\\\
\\hline
"""
        
        for row in metrics_data:
            latex_table += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\end{table}
"""
        
        # Save to file
        output_file = self.output_dir / 'detailed_metrics_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_table)
        
        logger.success(f"{GREEN}Detailed metrics table saved to {output_file}{END}")
        return str(output_file)

    def create_experiment_summary_table(self) -> str:
        """Creates an experiment summary table."""
        
        summary = self.results['experiment_summary']
        
        # Create LaTeX table
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Experiment Summary Statistics}
\\label{tab:experiment_summary}
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Metric} & \\textbf{Value} \\\\
\\hline
Total Claims Processed & """ + str(summary['total_claims_processed']) + """ \\\\
Total Challenges Generated & """ + str(summary['total_challenges_generated']) + """ \\\\
Total Revisions Attempted & """ + str(summary['total_revisions_attempted']) + """ \\\\
Experiment Duration & """ + str(summary['experiment_duration']) + """ \\\\
\\hline
\\end{tabular}
\\end{table}
"""
        
        # Save to file
        output_file = self.output_dir / 'experiment_summary_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_table)
        
        logger.success(f"{GREEN}Experiment summary table saved to {output_file}{END}")
        return str(output_file)

    def create_system_architecture_diagram(self) -> str:
        """Creates a system architecture flow diagram."""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define components and their positions
        components = {
            'ClaimDataset': (2, 8),
            'Answering Agent 1': (1, 6),
            'Answering Agent 2': (3, 6),
            'Challenger Agent': (2, 4),
            'Revision Manager': (2, 2),
            'Evaluation': (2, 0)
        }
        
        # Draw components as boxes
        for comp, (x, y) in components.items():
            if 'Agent' in comp:
                color = '#FFD93D'  # Yellow for agents
                if 'Challenger' in comp:
                    color = '#FF6B6B'  # Red for challenger
            elif comp == 'ClaimDataset':
                color = '#6BCF7F'  # Green for data
            elif comp == 'Evaluation':
                color = '#4D96FF'  # Blue for evaluation
            else:
                color = '#9B59B6'  # Purple for processing
            
            rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                               facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, comp, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw arrows showing data flow
        arrows = [
            ((2, 7.7), (1, 6.3)),  # Dataset to Agent 1
            ((2, 7.7), (3, 6.3)),  # Dataset to Agent 2
            ((1, 5.7), (2, 4.3)),  # Agent 1 to Challenger
            ((3, 5.7), (2, 4.3)),  # Agent 2 to Challenger
            ((2, 3.7), (2, 2.3)),  # Challenger to Revision
            ((2, 1.7), (2, 0.3)),  # Revision to Evaluation
        ]
        
        for (x1, y1), (x2, y2) in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        # Add title and labels
        ax.set_title('Cite-and-Challenge System Architecture', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 4)
        ax.set_ylim(-1, 9)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#6BCF7F', alpha=0.7, label='Data Layer'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FFD93D', alpha=0.7, label='Answering Agents'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', alpha=0.7, label='Challenger Agent'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#9B59B6', alpha=0.7, label='Processing'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#4D96FF', alpha=0.7, label='Evaluation')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'system_architecture.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success(f"{GREEN}System architecture diagram saved to {output_file}{END}")
        return str(output_file)

    def generate_all_figures(self) -> List[str]:
        """Generates all figures and tables for the paper."""
        
        logger.info(f"{BLUE}Starting figure generation process{END}")
        
        generated_files = []
        
        try:
            # Generate charts
            generated_files.append(self.create_performance_comparison_chart())
            generated_files.append(self.create_metrics_radar_chart())
            generated_files.append(self.create_improvement_analysis_chart())
            generated_files.append(self.create_system_architecture_diagram())
            
            # Generate tables
            generated_files.append(self.create_detailed_metrics_table())
            generated_files.append(self.create_experiment_summary_table())
            
            logger.success(f"{GREEN}All figures and tables generated successfully!{END}")
            logger.info(f"{CYAN}Generated files: {len(generated_files)}{END}")
            
            for file in generated_files:
                logger.info(f"  • {file}")
            
            return generated_files
            
        except Exception as e:
            logger.error(f"{RED}Error generating figures: {e}{END}")
            raise

def main():
    """Main function to generate all figures and tables."""
    
    # Find the most recent results file
    results_dir = Path('results')
    if not results_dir.exists():
        logger.error(f"{RED}Results directory not found{END}")
        return
    
    result_files = list(results_dir.glob('experiment_results_*.json'))
    if not result_files:
        logger.error(f"{RED}No experiment results found{END}")
        return
    
    # Use the most recent results file
    latest_results = sorted(result_files, key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"{BLUE}Using results file: {latest_results}{END}")
    
    # Generate figures
    generator = FigureGenerator(str(latest_results))
    generated_files = generator.generate_all_figures()
    
    logger.success(f"{GREEN}Figure generation complete! Generated {len(generated_files)} files.{END}")

if __name__ == "__main__":
    main()