"""
Visualization cho Ragas evaluation metrics
Tạo charts và graphs để visualize kết quả đánh giá
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
# Support Vietnamese characters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class RagasVisualizer:
    """Visualize Ragas evaluation results"""
    
    def __init__(self, report_path: Path):
        """Load evaluation report"""
        with open(report_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.output_dir = report_path.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_metrics_summary(self):
        """Plot tổng quan các metrics"""
        if 'metrics' not in self.data:
            print("Không tìm thấy metrics trong report")
            return
        
        metrics = self.data['metrics']
        total_questions = self.data.get('total_questions', 'N/A')
        
        # Translate metric names to Vietnamese
        metric_translation = {
            'faithfulness': 'Độ trung thực',
            'answer_relevancy': 'Độ liên quan của câu trả lời',
            'context_precision': 'Độ chính xác của ngữ cảnh',
            'context_recall': 'Độ bao phủ của ngữ cảnh'
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        metric_names = [metric_translation.get(k, k) for k in metrics.keys()]
        metric_values = list(metrics.values())
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax1.set_ylabel('Điểm số', fontsize=12)
        ax1.set_title(f'Tổng Quan Các Chỉ Số Đánh Giá\n(Đánh giá trên {total_questions} câu hỏi)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.axhline(y=0.8, color='green', linestyle='--', label='Xuất sắc (>0.8)', alpha=0.5)
        ax1.axhline(y=0.6, color='orange', linestyle='--', label='Tốt (>0.6)', alpha=0.5)
        ax1.legend(fontsize=10)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        values = metric_values + [metric_values[0]]  # Close the circle
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax2.fill(angles, values, alpha=0.25, color='#3498db')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names, fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.set_title(f'Biểu Đồ Radar Các Chỉ Số\n(Đánh giá trên {total_questions} câu hỏi)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / f'metrics_summary_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved metrics summary to: {output_path}")
        return output_path
    
    def plot_metrics_comparison(self, threshold_excellent=0.8, threshold_good=0.6):
        """Plot so sánh metrics với thresholds"""
        if 'metrics' not in self.data:
            return
        
        metrics = self.data['metrics']
        total_questions = self.data.get('total_questions', 'N/A')
        
        # Translate metric names to Vietnamese
        metric_translation = {
            'faithfulness': 'Độ trung thực',
            'answer_relevancy': 'Độ liên quan của câu trả lời',
            'context_precision': 'Độ chính xác của ngữ cảnh',
            'context_recall': 'Độ bao phủ của ngữ cảnh'
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = [metric_translation.get(k, k) for k in metrics.keys()]
        metric_values = list(metrics.values())
        
        # Create horizontal bar chart
        y_pos = np.arange(len(metric_names))
        
        # Color based on performance
        colors = []
        for val in metric_values:
            if val >= threshold_excellent:
                colors.append('#2ecc71')  # Green
            elif val >= threshold_good:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        bars = ax.barh(y_pos, metric_values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Điểm số', fontsize=12)
        ax.set_title(f'Đánh Giá Hiệu Suất Các Chỉ Số\n(Đánh giá trên {total_questions} câu hỏi)', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add threshold lines
        ax.axvline(x=threshold_excellent, color='green', linestyle='--', 
                  label=f'Xuất sắc (≥{threshold_excellent})', alpha=0.5)
        ax.axvline(x=threshold_good, color='orange', linestyle='--', 
                  label=f'Tốt (≥{threshold_good})', alpha=0.5)
        ax.legend(fontsize=10)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center')
        
        plt.tight_layout()
        output_path = self.output_dir / f'metrics_comparison_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved metrics comparison to: {output_path}")
        return output_path
    
    def plot_difficulty_breakdown(self, detailed_report_path: Path):
        """Plot breakdown theo difficulty level"""
        try:
            with open(detailed_report_path, 'r', encoding='utf-8') as f:
                detailed_data = json.load(f)
        except FileNotFoundError:
            print(f"Không tìm thấy detailed report: {detailed_report_path}")
            return
        
        total_questions = len(detailed_data)
        
        # Translate difficulty levels
        difficulty_translation = {
            'easy': 'Dễ',
            'medium': 'Trung bình',
            'hard': 'Khó',
            'unknown': 'Chưa xác định'
        }
        
        # Count by difficulty
        difficulty_count = {}
        for item in detailed_data:
            diff = item.get('difficulty', 'unknown')
            difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c', 'unknown': '#95a5a6'}
        
        # Translate labels for display
        translated_labels = [difficulty_translation.get(k, k) for k in difficulty_count.keys()]
        
        wedges, texts, autotexts = ax.pie(
            difficulty_count.values(),
            labels=translated_labels,
            colors=[colors.get(k, '#95a5a6') for k in difficulty_count.keys()],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11}
        )
        
        ax.set_title(f'Phân Bố Độ Khó Của Câu Hỏi Kiểm Tra\n(Tổng số: {total_questions} câu hỏi)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / f'difficulty_distribution_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved difficulty distribution to: {output_path}")
        return output_path
    
    def create_full_report(self, detailed_report_path: Path = None):
        """Tạo full report với tất cả visualizations"""
        print("\n" + "=" * 60)
        print("Đang Tạo Báo Cáo Trực Quan Hóa")
        print("=" * 60 + "\n")
        
        outputs = []
        
        # Metrics summary
        output = self.plot_metrics_summary()
        if output:
            outputs.append(output)
        
        # Metrics comparison
        output = self.plot_metrics_comparison()
        if output:
            outputs.append(output)
        
        # Difficulty breakdown
        if detailed_report_path and detailed_report_path.exists():
            output = self.plot_difficulty_breakdown(detailed_report_path)
            if output:
                outputs.append(output)
        
        print("\n" + "=" * 60)
        print(f"✓ Đã tạo {len(outputs)} biểu đồ trực quan")
        print("=" * 60 + "\n")
        
        return outputs


def main():
    """Main function"""
    import sys
    from glob import glob
    
    # Find latest report
    reports_dir = Path(__file__).parent.parent / "evaluation" / "reports"
    
    if not reports_dir.exists():
        print(f"Không tìm thấy thư mục reports: {reports_dir}")
        sys.exit(1)
    
    # Find latest summary report
    summary_files = sorted(glob(str(reports_dir / "ragas_summary_*.json")), reverse=True)
    
    if not summary_files:
        print("Không tìm thấy báo cáo đánh giá nào!")
        print("Hãy chạy 'python tests/evaluation/run_ragas_evaluation.py' trước")
        sys.exit(1)
    
    latest_summary = Path(summary_files[0])
    print(f"Đang sử dụng báo cáo: {latest_summary.name}")
    
    # Find corresponding detailed report
    timestamp = latest_summary.stem.replace('ragas_summary_', '')
    detailed_report = reports_dir / f"ragas_detailed_{timestamp}.json"
    
    # Create visualizations
    visualizer = RagasVisualizer(latest_summary)
    visualizer.create_full_report(detailed_report if detailed_report.exists() else None)


if __name__ == "__main__":
    main()
