import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import json

class PositionConfig:
    """位置标记配置类"""
    def __init__(self):
        # 可视化参数
        self.figure_size = (12, 10)
        self.dpi = 300
        self.point_size = 100
        self.point_alpha = 0.8
        self.text_size = 10
        
        # 颜色设置
        self.default_color = 'red'
        self.highlight_color = 'blue'
        self.text_color = 'black'
        
        # 网格设置
        self.grid_alpha = 0.3
        self.grid_color = 'gray'

class PositionManager:
    """
    位置管理器：处理神经元位置数据的存储、加载和可视化
    """
    
    def __init__(self, config: PositionConfig = None):
        self.config = config or PositionConfig()
        self.positions = {}  # {neuron_id: (x, y)}
        self.next_number = 1
        
    def add_position(self, neuron_id: str, x: float, y: float) -> None:
        """添加神经元位置"""
        self.positions[neuron_id] = (x, y)
    
    def remove_position(self, neuron_id: str) -> None:
        """删除神经元位置"""
        if neuron_id in self.positions:
            del self.positions[neuron_id]
    
    def update_position(self, neuron_id: str, x: float, y: float) -> None:
        """更新神经元位置"""
        if neuron_id in self.positions:
            self.positions[neuron_id] = (x, y)
    
    def get_position(self, neuron_id: str) -> Optional[Tuple[float, float]]:
        """获取神经元位置"""
        return self.positions.get(neuron_id)
    
    def get_all_positions(self) -> Dict[str, Tuple[float, float]]:
        """获取所有位置"""
        return self.positions.copy()
    
    def load_positions_from_dataframe(self, df: pd.DataFrame) -> None:
        """从DataFrame加载位置数据"""
        if 'number' in df.columns and 'relative_x' in df.columns and 'relative_y' in df.columns:
            for _, row in df.iterrows():
                neuron_id = f"Neuron_{int(row['number'])}"
                x = float(row['relative_x'])
                y = float(row['relative_y'])
                self.add_position(neuron_id, x, y)
        else:
            raise ValueError("DataFrame must contain 'number', 'relative_x', and 'relative_y' columns")
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = []
        for neuron_id, (x, y) in self.positions.items():
            # 提取神经元编号
            number = int(neuron_id.split('_')[1]) if '_' in neuron_id else len(data) + 1
            data.append({
                'number': number,
                'neuron_id': neuron_id,
                'relative_x': x,
                'relative_y': y
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('number')
    
    def create_position_plot(self, 
                           highlight_neurons: List[str] = None,
                           title: str = "Neuron Positions") -> str:
        """
        创建位置可视化图
        """
        if not self.positions:
            # 创建空图
            plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
            plt.text(0.5, 0.5, 'No position data available', 
                    ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
            plt.title(title)
        else:
            plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
            
            # 提取坐标
            x_coords = [pos[0] for pos in self.positions.values()]
            y_coords = [pos[1] for pos in self.positions.values()]
            neuron_ids = list(self.positions.keys())
            
            # 绘制所有神经元
            plt.scatter(x_coords, y_coords, 
                       s=self.config.point_size, 
                       alpha=self.config.point_alpha,
                       c=self.config.default_color,
                       edgecolors='black',
                       linewidth=1)
            
            # 高亮特定神经元
            if highlight_neurons:
                highlight_x = []
                highlight_y = []
                for neuron_id in highlight_neurons:
                    if neuron_id in self.positions:
                        x, y = self.positions[neuron_id]
                        highlight_x.append(x)
                        highlight_y.append(y)
                
                if highlight_x:
                    plt.scatter(highlight_x, highlight_y,
                               s=self.config.point_size * 1.5,
                               alpha=self.config.point_alpha,
                               c=self.config.highlight_color,
                               edgecolors='black',
                               linewidth=2,
                               label='Key Neurons')
            
            # 添加神经元标签
            for neuron_id, (x, y) in self.positions.items():
                plt.annotate(neuron_id, (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.config.text_size,
                           color=self.config.text_color)
            
            # 设置图形属性
            plt.xlabel('Relative X Position')
            plt.ylabel('Relative Y Position')
            plt.title(title)
            plt.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)
            plt.legend()
            
            # 设置坐标轴范围
            if x_coords and y_coords:
                x_margin = (max(x_coords) - min(x_coords)) * 0.1
                y_margin = (max(y_coords) - min(y_coords)) * 0.1
                plt.xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
                plt.ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def create_behavior_overlay_plot(self, 
                                   key_neurons_by_behavior: Dict[str, List[str]],
                                   title: str = "Key Neurons by Behavior") -> str:
        """
        创建按行为分类的关键神经元位置图
        """
        plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        if not self.positions:
            plt.text(0.5, 0.5, 'No position data available', 
                    ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
            plt.title(title)
        else:
            # 定义颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # 绘制所有神经元（灰色背景）
            all_x = [pos[0] for pos in self.positions.values()]
            all_y = [pos[1] for pos in self.positions.values()]
            plt.scatter(all_x, all_y, 
                       s=self.config.point_size * 0.5, 
                       alpha=0.3,
                       c='lightgray',
                       edgecolors='black',
                       linewidth=0.5)
            
            # 为每种行为绘制关键神经元
            legend_elements = []
            for i, (behavior, neurons) in enumerate(key_neurons_by_behavior.items()):
                if i >= len(colors):
                    color = colors[i % len(colors)]
                else:
                    color = colors[i]
                
                behavior_x = []
                behavior_y = []
                
                for neuron_id in neurons:
                    if neuron_id in self.positions:
                        x, y = self.positions[neuron_id]
                        behavior_x.append(x)
                        behavior_y.append(y)
                
                if behavior_x:
                    plt.scatter(behavior_x, behavior_y,
                               s=self.config.point_size,
                               alpha=self.config.point_alpha,
                               c=color,
                               edgecolors='black',
                               linewidth=1,
                               label=f'{behavior} ({len(behavior_x)} neurons)')
                    
                    # 添加神经元标签
                    for neuron_id in neurons:
                        if neuron_id in self.positions:
                            x, y = self.positions[neuron_id]
                            plt.annotate(neuron_id, (x, y), 
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=self.config.text_size - 2,
                                       color=self.config.text_color)
            
            plt.xlabel('Relative X Position')
            plt.ylabel('Relative Y Position')
            plt.title(title)
            plt.grid(True, alpha=self.config.grid_alpha, color=self.config.grid_color)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"

def process_position_data(positions_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理位置数据
    """
    config = PositionConfig()
    manager = PositionManager(config)
    
    # 从数据中加载位置信息
    if 'positions' in positions_data:
        for neuron_id, pos in positions_data['positions'].items():
            manager.add_position(neuron_id, pos['x'], pos['y'])
    
    # 生成位置图
    position_plot = manager.create_position_plot()
    
    # 如果有关键神经元信息，生成叠加图
    behavior_overlay_plot = None
    if 'key_neurons' in positions_data:
        behavior_overlay_plot = manager.create_behavior_overlay_plot(
            positions_data['key_neurons']
        )
    
    return {
        'position_plot': position_plot,
        'behavior_overlay_plot': behavior_overlay_plot,
        'positions_dataframe': manager.to_dataframe().to_dict('records'),
        'total_neurons': len(manager.positions)
    }
