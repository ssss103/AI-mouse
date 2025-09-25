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
        # 检查神经元数据中的NaN值 - 确保数据类型为数值型
        try:
            neuron_data_float = neuron_data.astype(float)
            neuron_nan_mask = np.isnan(neuron_data_float).any(axis=1)
        except (ValueError, TypeError):
            # 如果无法转换为float，则检查是否有非数值数据
            neuron_nan_mask = np.array([False] * len(neuron_data))
        
        # 检查行为数据中的NaN值 - 确保数据类型为数值型
        try:
            behavior_data_float = behavior_data.astype(float)
            behavior_nan_mask = np.isnan(behavior_data_float)
        except (ValueError, TypeError):
            # 如果无法转换为float，则检查是否有非数值数据
            behavior_nan_mask = np.array([False] * len(behavior_data))
        
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
        
        # 确保数据是数值类型
        try:
            group1 = group1.astype(float)
            group2 = group2.astype(float)
        except (ValueError, TypeError):
            return 0.0
        
        # 计算均值和标准差
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        # 计算标准差，处理样本数量不足的情况
        n1, n2 = len(group1), len(group2)
        if n1 <= 1:
            std1 = 0.0
        else:
            std1 = np.std(group1, ddof=1)
            
        if n2 <= 1:
            std2 = 0.0
        else:
            std2 = np.std(group2, ddof=1)
        
        # 计算合并标准差
        if n1 + n2 <= 2:
            pooled_std = 0.0
        else:
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
        if not effect_sizes:
            # 如果没有效应量数据，返回空的DataFrame
            return pd.DataFrame()
        
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
        
        # 检查DataFrame是否为空
        if effect_sizes_df.empty:
            print("Debug: effect_sizes_df is empty")
            return key_neurons
        
        print(f"Debug: effect_sizes_df columns: {effect_sizes_df.columns.tolist()}")
        print(f"Debug: effect_sizes_df shape: {effect_sizes_df.shape}")
        
        for behavior in effect_sizes_df.columns:
            try:
                # 找出效应量超过阈值的神经元
                significant_neurons = effect_sizes_df[
                    effect_sizes_df[behavior].abs() >= threshold
                ].index.tolist()
                
                print(f"Debug: {behavior} -> significant_neurons type: {type(significant_neurons)}, value: {significant_neurons}")
                
                # 确保返回的是列表
                key_neurons[behavior] = significant_neurons if isinstance(significant_neurons, list) else []
            except Exception as e:
                print(f"处理行为 {behavior} 时出错: {e}")
                key_neurons[behavior] = []
        
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
    
    # 检查效应量数据是否为空
    if effect_sizes_df.empty:
        return {
            'effect_sizes': {},
            'key_neurons': {},
            'statistics': {
                'total_neurons': 0,
                'total_behaviors': 0,
                'nan_info': calculator.nan_info,
                'threshold_used': threshold,
                'key_neurons_count': {}
            },
            'histogram_image': None,
            'heatmap_image': None
        }
    
    # 识别关键神经元
    print(f"Debug: 开始识别关键神经元，threshold: {threshold}")
    key_neurons = calculator.identify_key_neurons(effect_sizes_df, threshold)
    print(f"Debug: 关键神经元识别完成，结果: {key_neurons}")
    
    # 生成可视化
    histogram_image = calculator.create_effect_size_histogram(effect_sizes_df)
    heatmap_image = calculator.create_effect_size_heatmap(effect_sizes_df)
    
    # 统计信息
    stats = {
        'total_neurons': len(effect_sizes_df),
        'total_behaviors': len(effect_sizes_df.columns),
        'nan_info': calculator.nan_info,
        'threshold_used': threshold,
        'key_neurons_count': {behavior: len(neurons) if isinstance(neurons, (list, tuple)) else 0 for behavior, neurons in key_neurons.items()}
    }
    
    # 确保所有numpy类型都转换为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    return {
        'effect_sizes': convert_numpy_types(effect_sizes_df.to_dict()),
        'key_neurons': convert_numpy_types(key_neurons),
        'statistics': convert_numpy_types(stats),
        'histogram_image': histogram_image,
        'heatmap_image': heatmap_image
    }
