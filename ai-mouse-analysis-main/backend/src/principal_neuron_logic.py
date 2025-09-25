import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from src.effect_size_logic import EffectSizeCalculator, EffectSizeConfig
from src.position_logic import PositionManager, PositionConfig

class PrincipalNeuronConfig:
    """主神经元分析配置类"""
    def __init__(self):
        # 效应量阈值
        self.effect_size_threshold = 0.5
        self.small_effect = 0.2
        self.medium_effect = 0.5
        self.large_effect = 0.8
        
        # 可视化参数
        self.figure_size = (15, 10)
        self.dpi = 300
        self.point_size = 100
        self.text_size = 8
        
        # 颜色设置
        self.behavior_colors = {
            'Close': 'red',
            'Middle': 'blue', 
            'Open': 'green',
            'Explore': 'orange',
            'Rest': 'purple',
            'Groom': 'brown',
            'Climb': 'pink'
        }

class PrincipalNeuronAnalyzer:
    """
    主神经元分析器：整合效应量分析和位置可视化
    """
    
    def __init__(self, config: PrincipalNeuronConfig = None):
        self.config = config or PrincipalNeuronConfig()
        self.effect_calculator = EffectSizeCalculator(EffectSizeConfig())
        self.position_manager = PositionManager(PositionConfig())
        
    def analyze_shared_neurons(self, key_neurons: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        分析神经元之间的共享关系
        """
        behaviors = list(key_neurons.keys())
        shared_analysis = {}
        
        # 分析每对行为之间的共享神经元
        for behavior1, behavior2 in combinations(behaviors, 2):
            neurons1 = set(key_neurons[behavior1])
            neurons2 = set(key_neurons[behavior2])
            
            shared = neurons1.intersection(neurons2)
            unique1 = neurons1 - neurons2
            unique2 = neurons2 - neurons1
            
            shared_analysis[f"{behavior1}_and_{behavior2}"] = {
                'shared': list(shared),
                'unique_to_first': list(unique1),
                'unique_to_second': list(unique2),
                'shared_count': len(shared),
                'unique_first_count': len(unique1),
                'unique_second_count': len(unique2)
            }
        
        # 分析所有行为共享的神经元
        if len(behaviors) > 2:
            all_shared = set(key_neurons[behaviors[0]])
            for behavior in behaviors[1:]:
                all_shared = all_shared.intersection(set(key_neurons[behavior]))
            
            shared_analysis['all_behaviors'] = {
                'shared': list(all_shared),
                'shared_count': len(all_shared)
            }
        
        return shared_analysis
    
    def create_network_analysis_plot(self, 
                                   key_neurons: Dict[str, List[str]],
                                   shared_analysis: Dict[str, Any]) -> str:
        """
        创建网络分析图
        """
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # 创建子图
        n_behaviors = len(key_neurons)
        if n_behaviors <= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        
        # 1. 关键神经元数量统计
        ax1 = axes[0]
        behaviors = list(key_neurons.keys())
        counts = [len(neurons) for neurons in key_neurons.values()]
        colors = [self.config.behavior_colors.get(behavior, 'gray') for behavior in behaviors]
        
        bars = ax1.bar(behaviors, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Key Neurons Count by Behavior')
        ax1.set_ylabel('Number of Key Neurons')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 2. 共享关系分析
        if len(behaviors) >= 2:
            ax2 = axes[1]
            shared_pairs = []
            shared_counts = []
            
            for key, data in shared_analysis.items():
                if 'shared_count' in data and key != 'all_behaviors':
                    shared_pairs.append(key.replace('_and_', ' & '))
                    shared_counts.append(data['shared_count'])
            
            if shared_pairs:
                bars2 = ax2.bar(shared_pairs, shared_counts, color='lightblue', alpha=0.7, edgecolor='black')
                ax2.set_title('Shared Neurons Between Behaviors')
                ax2.set_ylabel('Number of Shared Neurons')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar, count in zip(bars2, shared_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
        
        # 3. 效应量分布（如果有数据）
        if len(axes) > 2:
            ax3 = axes[2]
            # 这里可以添加效应量分布图
            ax3.text(0.5, 0.5, 'Effect Size Distribution\n(Data Required)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Effect Size Distribution')
        
        # 4. 网络连接图（如果有数据）
        if len(axes) > 3:
            ax4 = axes[3]
            ax4.text(0.5, 0.5, 'Network Connectivity\n(Data Required)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Neuron Network')
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def create_comprehensive_analysis_plot(self,
                                         effect_sizes_df: pd.DataFrame,
                                         key_neurons: Dict[str, List[str]],
                                         positions_data: Dict[str, Any] = None) -> str:
        """
        创建综合分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 效应量热力图
        ax1 = axes[0, 0]
        sns.heatmap(effect_sizes_df.T, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', ax=ax1, cbar_kws={'label': 'Cohen\'s d'})
        ax1.set_title('Effect Sizes: Neurons vs Behaviors')
        ax1.set_xlabel('Neuron ID')
        ax1.set_ylabel('Behavior')
        
        # 2. 关键神经元统计
        ax2 = axes[0, 1]
        behaviors = list(key_neurons.keys())
        counts = [len(neurons) for neurons in key_neurons.values()]
        colors = [self.config.behavior_colors.get(behavior, 'gray') for behavior in behaviors]
        
        bars = ax2.bar(behaviors, counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Key Neurons Count by Behavior')
        ax2.set_ylabel('Number of Key Neurons')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 3. 位置分布图（如果有位置数据）
        ax3 = axes[1, 0]
        if positions_data and 'positions' in positions_data:
            # 绘制所有神经元位置
            all_x = []
            all_y = []
            for pos in positions_data['positions'].values():
                all_x.append(pos['x'])
                all_y.append(pos['y'])
            
            ax3.scatter(all_x, all_y, s=50, alpha=0.3, c='lightgray', edgecolors='black')
            
            # 高亮关键神经元
            for behavior, neurons in key_neurons.items():
                color = self.config.behavior_colors.get(behavior, 'gray')
                behavior_x = []
                behavior_y = []
                
                for neuron_id in neurons:
                    if neuron_id in positions_data['positions']:
                        pos = positions_data['positions'][neuron_id]
                        behavior_x.append(pos['x'])
                        behavior_y.append(pos['y'])
                
                if behavior_x:
                    ax3.scatter(behavior_x, behavior_y, s=100, alpha=0.8, 
                               c=color, edgecolors='black', label=f'{behavior} ({len(behavior_x)})')
            
            ax3.set_xlabel('Relative X Position')
            ax3.set_ylabel('Relative Y Position')
            ax3.set_title('Key Neurons Spatial Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Position Data Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Spatial Distribution')
        
        # 4. 效应量分布直方图
        ax4 = axes[1, 1]
        all_effect_sizes = []
        for col in effect_sizes_df.columns:
            all_effect_sizes.extend(effect_sizes_df[col].values)
        
        ax4.hist(all_effect_sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(self.config.small_effect, color='orange', linestyle='--', 
                   label=f'Small Effect ({self.config.small_effect})')
        ax4.axvline(self.config.medium_effect, color='red', linestyle='--', 
                   label=f'Medium Effect ({self.config.medium_effect})')
        ax4.axvline(self.config.large_effect, color='darkred', linestyle='--', 
                   label=f'Large Effect ({self.config.large_effect})')
        ax4.axvline(-self.config.small_effect, color='orange', linestyle='--')
        ax4.axvline(-self.config.medium_effect, color='red', linestyle='--')
        ax4.axvline(-self.config.large_effect, color='darkred', linestyle='--')
        
        ax4.set_xlabel('Cohen\'s d Effect Size')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Effect Size Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_analysis_report(self,
                               effect_sizes_df: pd.DataFrame,
                               key_neurons: Dict[str, List[str]],
                               shared_analysis: Dict[str, Any],
                               positions_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成完整的分析报告
        """
        # 基础统计
        total_neurons = len(effect_sizes_df)
        total_behaviors = len(effect_sizes_df.columns)
        total_key_neurons = sum(len(neurons) for neurons in key_neurons.values())
        
        # 效应量统计
        all_effect_sizes = []
        for col in effect_sizes_df.columns:
            all_effect_sizes.extend(effect_sizes_df[col].values)
        
        effect_size_stats = {
            'mean': np.mean(all_effect_sizes),
            'std': np.std(all_effect_sizes),
            'min': np.min(all_effect_sizes),
            'max': np.max(all_effect_sizes),
            'median': np.median(all_effect_sizes)
        }
        
        # 生成可视化
        network_plot = self.create_network_analysis_plot(key_neurons, shared_analysis)
        comprehensive_plot = self.create_comprehensive_analysis_plot(
            effect_sizes_df, key_neurons, positions_data
        )
        
        return {
            'summary': {
                'total_neurons': total_neurons,
                'total_behaviors': total_behaviors,
                'total_key_neurons': total_key_neurons,
                'key_neurons_per_behavior': {behavior: len(neurons) for behavior, neurons in key_neurons.items()}
            },
            'effect_size_statistics': effect_size_stats,
            'shared_analysis': shared_analysis,
            'visualizations': {
                'network_analysis': network_plot,
                'comprehensive_analysis': comprehensive_plot
            },
            'key_neurons': key_neurons,
            'effect_sizes': effect_sizes_df.to_dict()
        }

def analyze_principal_neurons(data: pd.DataFrame,
                            behavior_column: str = None,
                            positions_data: Dict[str, Any] = None,
                            threshold: float = 0.5) -> Dict[str, Any]:
    """
    完整的主神经元分析流程
    """
    config = PrincipalNeuronConfig()
    config.effect_size_threshold = threshold
    
    analyzer = PrincipalNeuronAnalyzer(config)
    
    # 1. 计算效应量
    effect_sizes_df = analyzer.effect_calculator.calculate_effect_sizes(data, behavior_column)
    
    # 2. 识别关键神经元
    key_neurons = analyzer.effect_calculator.identify_key_neurons(effect_sizes_df, threshold)
    
    # 3. 分析共享关系
    shared_analysis = analyzer.analyze_shared_neurons(key_neurons)
    
    # 4. 生成完整报告
    report = analyzer.generate_analysis_report(
        effect_sizes_df, key_neurons, shared_analysis, positions_data
    )
    
    return report
