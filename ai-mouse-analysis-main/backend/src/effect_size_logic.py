import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

class EffectSizeConfig:
    """效应量分析配置类"""
    def __init__(self):
        # 效应量阈值设置
        self.effect_size_threshold = 0.5  # Cohen's d阈值
        self.small_effect = 0.2
        self.medium_effect = 0.5
        self.large_effect = 0.8
        
        # 统计参数
        self.confidence_level = 0.95
        self.random_state = 42
        
        # 可视化参数
        self.figure_size = (12, 8)
        self.dpi = 300

class EffectSizeCalculator:
    """
    效应量计算器类：用于计算神经元活动与行为之间的Cohen's d效应量
    """
    
    def __init__(self, config: EffectSizeConfig = None):
        self.config = config or EffectSizeConfig()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.nan_info = {}
        
    def remove_nan_rows(self, neuron_data: np.ndarray, behavior_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测并删除包含NaN值的行
        """
        # 检查神经元数据中的NaN值
        neuron_nan_mask = np.isnan(neuron_data).any(axis=1)
        
        # 检查行为数据中的NaN值
        behavior_nan_mask = np.isnan(behavior_data)
        
        # 合并NaN掩码
        combined_nan_mask = neuron_nan_mask | behavior_nan_mask
        
        # 记录NaN信息
        self.nan_info = {
            'total_rows': len(neuron_data),
            'nan_rows': np.sum(combined_nan_mask),
            'neuron_nan_rows': np.sum(neuron_nan_mask),
            'behavior_nan_rows': np.sum(behavior_nan_mask)
        }
        
        # 删除包含NaN的行
        if np.any(combined_nan_mask):
            cleaned_neuron_data = neuron_data[~combined_nan_mask]
            cleaned_behavior_data = behavior_data[~combined_nan_mask]
            return cleaned_neuron_data, cleaned_behavior_data
        
        return neuron_data, behavior_data
    
    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        计算Cohen's d效应量
        """
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        # 计算均值和标准差
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # 计算合并标准差
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        # 计算Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
    
    def calculate_effect_sizes(self, data: pd.DataFrame, behavior_column: str = None) -> pd.DataFrame:
        """
        计算所有神经元对所有行为的效应量
        """
        if behavior_column is None:
            behavior_column = data.columns[-1]  # 使用最后一列作为行为列
        
        # 分离神经元数据和行为数据
        neuron_columns = [col for col in data.columns if col != behavior_column]
        neuron_data = data[neuron_columns].values
        behavior_data = data[behavior_column].values
        
        # 删除包含NaN的行
        neuron_data, behavior_data = self.remove_nan_rows(neuron_data, behavior_data)
        
        # 编码行为标签
        encoded_behaviors = self.label_encoder.fit_transform(behavior_data)
        unique_behaviors = self.label_encoder.classes_
        
        # 计算效应量矩阵
        effect_sizes = {}
        
        for i, behavior in enumerate(unique_behaviors):
            behavior_mask = encoded_behaviors == i
            other_mask = ~behavior_mask
            
            behavior_neurons = neuron_data[behavior_mask]
            other_neurons = neuron_data[other_mask]
            
            effect_sizes[behavior] = []
            
            for j in range(neuron_data.shape[1]):
                neuron_name = neuron_columns[j]
                cohens_d = self.calculate_cohens_d(
                    behavior_neurons[:, j], 
                    other_neurons[:, j]
                )
                effect_sizes[behavior].append(cohens_d)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(effect_sizes, index=neuron_columns)
        result_df.index.name = 'Neuron_ID'
        
        return result_df
    
    def identify_key_neurons(self, effect_sizes_df: pd.DataFrame, threshold: float = None) -> Dict[str, List[str]]:
        """
        识别关键神经元
        """
        if threshold is None:
            threshold = self.config.effect_size_threshold
        
        key_neurons = {}
        
        for behavior in effect_sizes_df.columns:
            # 找出效应量超过阈值的神经元
            significant_neurons = effect_sizes_df[
                effect_sizes_df[behavior].abs() >= threshold
            ].index.tolist()
            
            key_neurons[behavior] = significant_neurons
        
        return key_neurons
    
    def create_effect_size_histogram(self, effect_sizes_df: pd.DataFrame) -> str:
        """
        创建效应量分布直方图
        """
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # 收集所有效应量数据
        all_effect_sizes = []
        for col in effect_sizes_df.columns:
            all_effect_sizes.extend(effect_sizes_df[col].values)
        
        # 绘制直方图
        plt.hist(all_effect_sizes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加阈值线
        plt.axvline(self.config.small_effect, color='orange', linestyle='--', label=f'Small Effect ({self.config.small_effect})')
        plt.axvline(self.config.medium_effect, color='red', linestyle='--', label=f'Medium Effect ({self.config.medium_effect})')
        plt.axvline(self.config.large_effect, color='darkred', linestyle='--', label=f'Large Effect ({self.config.large_effect})')
        plt.axvline(-self.config.small_effect, color='orange', linestyle='--')
        plt.axvline(-self.config.medium_effect, color='red', linestyle='--')
        plt.axvline(-self.config.large_effect, color='darkred', linestyle='--')
        
        plt.xlabel('Cohen\'s d Effect Size')
        plt.ylabel('Frequency')
        plt.title('Distribution of Effect Sizes Across All Neurons and Behaviors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def create_effect_size_heatmap(self, effect_sizes_df: pd.DataFrame) -> str:
        """
        创建效应量热力图
        """
        plt.figure(figsize=(max(8, len(effect_sizes_df.columns) * 2), max(6, len(effect_sizes_df) * 0.3)), dpi=self.config.dpi)
        
        # 创建热力图
        sns.heatmap(effect_sizes_df.T, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.2f',
                   cbar_kws={'label': 'Cohen\'s d Effect Size'})
        
        plt.title('Effect Sizes: Neurons vs Behaviors')
        plt.xlabel('Neuron ID')
        plt.ylabel('Behavior')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

def analyze_effect_sizes(data: pd.DataFrame, 
                        behavior_column: str = None,
                        threshold: float = 0.5) -> Dict[str, Any]:
    """
    完整的效应量分析流程
    """
    config = EffectSizeConfig()
    config.effect_size_threshold = threshold
    
    calculator = EffectSizeCalculator(config)
    
    # 计算效应量
    effect_sizes_df = calculator.calculate_effect_sizes(data, behavior_column)
    
    # 识别关键神经元
    key_neurons = calculator.identify_key_neurons(effect_sizes_df, threshold)
    
    # 生成可视化
    histogram_image = calculator.create_effect_size_histogram(effect_sizes_df)
    heatmap_image = calculator.create_effect_size_heatmap(effect_sizes_df)
    
    # 统计信息
    stats = {
        'total_neurons': len(effect_sizes_df),
        'total_behaviors': len(effect_sizes_df.columns),
        'nan_info': calculator.nan_info,
        'threshold_used': threshold,
        'key_neurons_count': {behavior: len(neurons) for behavior, neurons in key_neurons.items()}
    }
    
    return {
        'effect_sizes': effect_sizes_df.to_dict(),
        'key_neurons': key_neurons,
        'statistics': stats,
        'histogram_image': histogram_image,
        'heatmap_image': heatmap_image
    }
