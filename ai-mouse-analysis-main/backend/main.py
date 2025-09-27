from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import shutil
import tempfile
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

# 导入核心逻辑模块
from src.extraction_logic import run_batch_extraction, extract_calcium_features, get_interactive_data, extract_manual_range
from src.clustering_logic import (
    load_data,
    enhance_preprocess_data,
    cluster_kmeans,
    visualize_clusters_2d,
    visualize_feature_distribution,
    analyze_clusters,
    generate_comprehensive_cluster_analysis,
    determine_optimal_k,
    cluster_dbscan,
    create_k_comparison_plot,
    plot_to_base64
)
from src.heatmap_behavior import (
    BehaviorHeatmapConfig,
    load_and_validate_data,
    find_behavior_pairs,
    extract_behavior_sequence_data,
    standardize_neural_data,
    create_behavior_sequence_heatmap,
    create_average_sequence_heatmap,
    get_global_neuron_order
)
from src.overall_heatmap import (
    OverallHeatmapConfig,
    generate_overall_heatmap
)
from src.heatmap_em_sort import (
    EMSortHeatmapConfig,
    analyze_em_sort_heatmap
)
from src.heatmap_multi_day import (
    MultiDayHeatmapConfig,
    analyze_multiday_heatmap
)
from src.trace_logic import (
    TraceConfig,
    load_trace_data,
    generate_trace_plot
)
from src.effect_size_analysis import analyze_effect_sizes
from src.position_marking import PositionMarker, process_position_data as process_position_data_v2, validate_position_data
from src.neuron_visualization import analyze_neuron_visualization
import numpy as np
import matplotlib.pyplot as plt
from src.utils import save_plot_as_base64
import base64
from io import BytesIO

# 配置matplotlib中文字体
def configure_chinese_font():
    """配置matplotlib中文字体"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 尝试设置中文字体
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'DejaVu Sans',  # 备用字体
            'Arial Unicode MS'  # 备用字体
        ]
        
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                break
            except:
                continue
        
        # 如果都失败了，使用默认设置
        if 'font.sans-serif' not in plt.rcParams or not plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
    except Exception as e:
        print(f"字体配置警告: {e}")
        # 使用默认设置
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化字体配置
configure_chinese_font()

def analyze_principal_neurons(data: pd.DataFrame, behavior_column: str = None, 
                            positions_dict: dict = None, threshold: float = 0.5) -> dict:
    """
    执行主神经元分析，结合效应量分析和位置数据
    
    参数
    ----------
    data : pd.DataFrame
        神经元活动数据
    behavior_column : str, 可选
        行为标签列名
    positions_dict : dict, 可选
        位置数据字典
    threshold : float
        效应量阈值
        
    返回
    ----------
    dict
        分析结果
    """
    try:
        # 1. 执行效应量分析
        effect_size_result = analyze_effect_sizes(data, behavior_column)
        
        # 2. 如果有位置数据，执行综合分析
        if positions_dict and positions_dict.get('positions'):
            # 将位置数据转换为DataFrame格式
            positions_data = []
            for neuron_id, pos in positions_dict['positions'].items():
                positions_data.append({
                    'neuron_id': neuron_id,
                    'x': pos.get('x', 0),
                    'y': pos.get('y', 0)
                })
            positions_df = pd.DataFrame(positions_data)
            
            # 将效应量数据转换为DataFrame格式
            effect_sizes_data = []
            for neuron_id, effects in effect_size_result['effect_sizes'].items():
                for behavior, effect_size in effects.items():
                    effect_sizes_data.append({
                        'neuron_id': neuron_id,
                        'behavior': behavior,
                        'effect_size': effect_size
                    })
            effect_sizes_df = pd.DataFrame(effect_sizes_data)
            
            # 执行神经元可视化分析
            visualization_result = analyze_neuron_visualization(
                effect_sizes_df, positions_df, threshold
            )
            
            # 合并结果，确保数据类型正确
            result = {
                'success': True,
                'effect_size_analysis': effect_size_result,
                'position_analysis': {
                    'total_positions': int(len(positions_data)),
                    'positions': positions_dict['positions']
                },
                'comprehensive_analysis': visualization_result,
                'statistics': {
                    'total_neurons': int(len(effect_size_result.get('effect_sizes', {}))),
                    'total_behaviors': int(len(effect_size_result.get('behavior_labels', []))),
                    'total_positions': int(len(positions_data)),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
        else:
            # 只有效应量分析
            result = {
                'success': True,
                'effect_size_analysis': effect_size_result,
                'statistics': {
                    'total_neurons': int(len(effect_size_result.get('effect_sizes', {}))),
                    'total_behaviors': int(len(effect_size_result.get('behavior_labels', []))),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
        
        return result
        
    except Exception as e:
        print(f"主神经元分析错误: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': '主神经元分析失败'
        }

app = FastAPI(title="钙信号分析平台 API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175", "http://127.0.0.1:5176"],  # Vue开发服务器地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 增加请求大小限制，解决431错误
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int = 100 * 1024 * 1024):  # 100MB
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_upload_size:
                return Response("File too large", status_code=413)
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware)

# 创建必要的目录
UPLOADS_DIR = Path("uploads")
RESULTS_DIR = Path("results")
TEMP_DIR = Path("temp")

for dir_path in [UPLOADS_DIR, RESULTS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "钙信号分析平台 API"}

@app.post("/api/extraction/preview")
async def preview_extraction(
    file: UploadFile = File(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(12),
    max_duration_frames: int = Form(800),
    min_snr: float = Form(3.5),
    smooth_window: int = Form(31),
    peak_distance_frames: int = Form(24),
    filter_strength: float = Form(1.0),
    neuron_id: str = Form(...)
):
    """预览单个神经元的事件提取结果"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 读取数据 - 支持多种文件格式
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_file)
        else:
            # 尝试自动检测文件格式
            try:
                df = pd.read_csv(temp_file)
            except:
                try:
                    df = pd.read_excel(temp_file)
                except:
                    raise ValueError(f"不支持的文件格式: {file.filename}")
        # 清理列名（去除可能的空格）
        df.columns = [col.strip() for col in df.columns]
        
        # 提取神经元列（以'n'开头且后面跟数字的列）
        neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
        
        # 如果neuron_id是'temp'，只返回神经元列表
        if neuron_id == 'temp':
            temp_file.unlink()
            return {
                "success": True,
                "neuron_columns": neuron_columns,
                "features": [],
                "plot": None
            }
        
        if neuron_id not in df.columns:
            raise HTTPException(status_code=400, detail=f"神经元 {neuron_id} 不存在")
        
        # 设置参数
        params = {
            'min_duration': min_duration_frames,
            'max_duration': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 提取特征并生成可视化
        feature_table, fig, _ = extract_calcium_features(
            df[neuron_id].values, fs=fs, visualize=True, params=params
        )
        
        # 将图表转换为base64
        plot_base64 = save_plot_as_base64(fig)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "features": feature_table.to_dict('records') if not feature_table.empty else [],
            "plot": plot_base64,
            "neuron_columns": neuron_columns
        }
        
    except Exception as e:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        print(f"Error in preview_extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/interactive_data")
async def get_interactive_extraction_data(
    file: UploadFile = File(...),
    neuron_id: str = Form(...)
):
    """获取交互式图表数据"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 获取交互式数据
        interactive_data = get_interactive_data(str(temp_file), neuron_id)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "data": interactive_data
        }
        
    except Exception as e:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/manual_extract")
async def manual_extraction(
    file: UploadFile = File(...),
    neuron_id: str = Form(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(5),
    max_duration_frames: int = Form(100),
    min_snr: float = Form(2.0),
    smooth_window: int = Form(5),
    peak_distance_frames: int = Form(10),
    filter_strength: float = Form(0.1)
):
    """基于用户选择的时间范围进行手动提取"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 构建参数字典
        params = {
            'fs': fs,
            'min_duration_frames': min_duration_frames,
            'max_duration_frames': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance_frames': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 执行手动提取
        result = extract_manual_range(str(temp_file), neuron_id, start_time, end_time, params)
        
        # 清理临时文件
        temp_file.unlink()
        
        return result
        
    except Exception as e:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/batch")
async def batch_extraction(
    files: List[UploadFile] = File(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(12),
    max_duration_frames: int = Form(800),
    min_snr: float = Form(3.5),
    smooth_window: int = Form(31),
    peak_distance_frames: int = Form(24),
    filter_strength: float = Form(1.0)
):
    """批量处理文件进行事件提取"""
    try:
        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        upload_dir = UPLOADS_DIR / timestamp
        upload_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件
        saved_file_paths = []
        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(str(file_path))
        
        # 设置参数
        params = {
            'min_duration': min_duration_frames,
            'max_duration': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 执行批量提取
        result_path = run_batch_extraction(saved_file_paths, str(RESULTS_DIR), fs=fs, **params)
        
        if result_path and os.path.exists(result_path):
            return {
                "success": True,
                "result_file": os.path.basename(result_path),
                "message": "批量分析完成"
            }
        else:
            raise HTTPException(status_code=500, detail="批量分析未生成任何结果")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/save_preview")
async def save_preview_result(
    data: str = Form(...)
):
    """保存单神经元预览结果"""
    try:
        # 解析前端传来的数据
        save_data = json.loads(data)
        
        # 构建输出文件名
        original_filename = save_data['filename']
        neuron = save_data['neuron']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = Path(original_filename).stem
        output_filename = f"{base_name}_{neuron}_features_{timestamp}.xlsx"
        output_path = RESULTS_DIR / output_filename
        
        # 构建DataFrame
        features_data = []
        for i, feature in enumerate(save_data['features']):
            feature_row = {
                'event_id': i + 1,
                'neuron': neuron,
                'amplitude': feature.get('amplitude', 0),
                'duration': feature.get('duration', 0),
                'fwhm': feature.get('fwhm', 0),
                'rise_time': feature.get('rise_time', 0),
                'decay_time': feature.get('decay_time', 0),
                'auc': feature.get('auc', 0),
                'snr': feature.get('snr', 0),
                'start_idx': feature.get('start_idx', 0),
                'peak_idx': feature.get('peak_idx', 0),
                'end_idx': feature.get('end_idx', 0),
                'start_time': feature.get('start_time', 0),
                'peak_time': feature.get('peak_time', 0),
                'end_time': feature.get('end_time', 0),
                'extraction_method': 'manual' if feature.get('isManualExtracted', False) else 'auto',
                'source_file': original_filename
            }
            features_data.append(feature_row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(features_data)
        df.to_excel(output_path, index=False)
        
        # 创建元数据
        metadata = {
            'original_file': original_filename,
            'neuron': neuron,
            'total_features': save_data['total_features'],
            'manual_features': save_data['manual_features'],
            'auto_features': save_data['auto_features'],
            'extraction_params': save_data['params'],
            'created_at': datetime.now().isoformat(),
            'file_type': 'single_neuron_preview'
        }
        
        # 保存元数据
        metadata_path = RESULTS_DIR / f"{base_name}_{neuron}_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True, 
            "filename": output_filename,
            "features_count": len(features_data),
            "message": f"成功保存 {len(features_data)} 个特征"
        }
        
    except Exception as e:
        print(f"保存预览结果错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")

@app.get("/api/results/files")
async def list_result_files():
    """获取结果文件列表"""
    try:
        feature_files = list(RESULTS_DIR.glob("*_features.xlsx"))
        files_info = []
        
        for file_path in feature_files:
            try:
                # 尝试从文件名解析时间戳
                basename = file_path.name
                timestamp_str = basename.split('_features.xlsx')[0].split('_')[-1]
                dt_obj = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
                friendly_name = f"{basename} (创建于: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')})"
            except (ValueError, IndexError):
                friendly_name = basename
            
            # 获取文件的创建时间
            stat = file_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            files_info.append({
                "filename": basename,
                "friendly_name": friendly_name,
                "path": str(file_path),
                "created_at": created_at,
                "size": stat.st_size
            })
        
        return {"success": True, "files": files_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clustering/analyze")
async def clustering_analysis(
    filename: str = Form(...),
    k: Optional[int] = Form(None),
    algorithm: str = Form("kmeans"),
    reduction_method: str = Form("pca"),
    feature_weights: Optional[str] = Form(None),
    auto_k: bool = Form(False),
    auto_k_range: str = Form("2,10"),
    dbscan_eps: float = Form(0.5),
    dbscan_min_samples: int = Form(5)
):
    """
    执行综合聚类分析
    
    参数:
    - filename: 数据文件名
    - k: K-means聚类数（如果为None且auto_k=True，则自动确定）
    - algorithm: 聚类算法 ('kmeans' 或 'dbscan')
    - reduction_method: 降维方法 ('pca' 或 'tsne')
    - feature_weights: JSON格式的特征权重字典
    - auto_k: 是否自动确定最佳K值
    - auto_k_range: 自动确定K值的搜索范围，格式 "min,max"
    - dbscan_eps: DBSCAN的eps参数
    - dbscan_min_samples: DBSCAN的min_samples参数
    """
    try:
        file_path = RESULTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 解析特征权重
        weights = None
        if feature_weights:
            try:
                weights = json.loads(feature_weights)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="特征权重格式错误，应为JSON格式")
        
        # 解析自动K值范围
        auto_k_min, auto_k_max = map(int, auto_k_range.split(','))
        
        # 如果启用自动K值且使用K-means，则将k设为None
        if auto_k and algorithm == 'kmeans':
            k = None
        
        # 执行综合聚类分析
        result = generate_comprehensive_cluster_analysis(
            file_path=str(file_path),
            k=k,
            algorithm=algorithm,
            feature_weights=weights,
            reduction_method=reduction_method,
            auto_k_range=(auto_k_min, auto_k_max),
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples
        )
        
        # 添加成功标志和请求参数
        result.update({
            "success": True,
            "request_params": {
                "filename": filename,
                "k": k,
                "algorithm": algorithm,
                "reduction_method": reduction_method,
                "feature_weights": weights,
                "auto_k": auto_k,
                "auto_k_range": (auto_k_min, auto_k_max)
            }
        })
        
        return result
        
    except Exception as e:
        print(f"聚类分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {str(e)}")

@app.post("/api/clustering/optimal_k")
async def find_optimal_k(
    filename: str = Form(...),
    max_k: int = Form(10),
    feature_weights: Optional[str] = Form(None)
):
    """
    确定最佳聚类数K
    """
    try:
        file_path = RESULTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 解析特征权重
        weights = None
        if feature_weights:
            try:
                weights = json.loads(feature_weights)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="特征权重格式错误，应为JSON格式")
        
        # 加载和预处理数据
        df = load_data(str(file_path))
        features_scaled, feature_names, df_clean = enhance_preprocess_data(df, weights)
        
        # 确定最佳K值
        optimal_k, inertia_values, silhouette_scores = determine_optimal_k(features_scaled, max_k)
        
        # 生成K值比较图
        from src.clustering_logic import create_optimal_k_plot
        k_range = list(range(2, max_k + 1))
        k_plot = create_optimal_k_plot(inertia_values, silhouette_scores, k_range)
        k_plot_base64 = plot_to_base64(k_plot)
        
        return {
            "success": True,
            "optimal_k": optimal_k,
            "k_range": k_range,
            "inertia_values": inertia_values,
            "silhouette_scores": silhouette_scores,
            "optimal_k_plot": k_plot_base64,
            "data_info": {
                "total_samples": len(df),
                "valid_samples": len(df_clean),
                "features_used": feature_names
            }
        }
        
    except Exception as e:
        print(f"最佳K值分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"最佳K值分析失败: {str(e)}")

@app.post("/api/clustering/compare_k")
async def compare_k_values(
    filename: str = Form(...),
    k_values: str = Form("2,3,4,5"),
    feature_weights: Optional[str] = Form(None)
):
    """
    比较不同K值的聚类效果
    """
    try:
        file_path = RESULTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 解析K值列表
        k_list = [int(k.strip()) for k in k_values.split(',')]
        
        # 解析特征权重
        weights = None
        if feature_weights:
            try:
                weights = json.loads(feature_weights)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="特征权重格式错误，应为JSON格式")
        
        # 加载和预处理数据
        df = load_data(str(file_path))
        features_scaled, feature_names, df_clean = enhance_preprocess_data(df, weights)
        
        # 创建K值比较图
        comparison_plot, silhouette_scores_dict = create_k_comparison_plot(features_scaled, k_list)
        comparison_plot_base64 = plot_to_base64(comparison_plot)
        
        # 找出最佳K值
        best_k = max(silhouette_scores_dict, key=silhouette_scores_dict.get)
        
        return {
            "success": True,
            "k_values": k_list,
            "silhouette_scores": silhouette_scores_dict,
            "best_k": best_k,
            "comparison_plot": comparison_plot_base64,
            "data_info": {
                "total_samples": len(df),
                "valid_samples": len(df_clean),
                "features_used": feature_names
            }
        }
        
    except Exception as e:
        print(f"K值比较错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"K值比较失败: {str(e)}")

@app.post("/api/behavior/detect")
async def detect_behavior_events(
    file: UploadFile = File(...)
):
    """检测行为事件配对"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"behavior_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 检测行为事件
        
        # 加载数据
        data = load_and_validate_data(str(temp_file))
        # 数据加载成功
        
        # 获取所有唯一的行为类型
        unique_behaviors = data['behavior'].unique().tolist()
        # 发现行为类型
        
        # 查找所有可能的行为配对（使用默认参数）
        behavior_events = []
        
        # 遍历所有可能的行为配对组合
        for start_behavior in unique_behaviors:
            for end_behavior in unique_behaviors:
                if start_behavior != end_behavior:
                    try:
                        pairs = find_behavior_pairs(
                            data, start_behavior, end_behavior, 
                            min_duration=1.0, sampling_rate=4.8
                        )
                        
                        for i, (start_begin, start_end, end_begin, end_end) in enumerate(pairs):
                            behavior_events.append({
                                'index': len(behavior_events) + 1,
                                'start_behavior': start_behavior,
                                'end_behavior': end_behavior,
                                'start_time': float(start_begin / 4.8),  # 转换为秒
                                'end_time': float(end_end / 4.8),  # 转换为秒
                                'duration': float((end_end - start_begin) / 4.8)  # 转换为秒
                            })
                    except Exception as e:
                        # 查找行为配对时出错
                        continue
        
        # 检测到行为事件配对
        
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        
        return {
            "success": True,
            "behavior_events": behavior_events,
            "available_behaviors": unique_behaviors,
            "message": f"检测到 {len(behavior_events)} 个行为事件配对"
        }
        
    except Exception as e:
        print(f"行为事件检测错误: {e}")
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"行为事件检测失败: {str(e)}")

@app.post("/api/heatmap/behaviors")
async def extract_behavior_labels(
    file: UploadFile = File(...)
):
    """
    提取上传文件中behavior列的唯一行为标签
    
    Parameters
    ----------
    file : UploadFile
        包含行为数据的Excel文件
        
    Returns
    -------
    dict
        包含behavior列唯一值的响应数据
    """
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"extract_behaviors_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 提取行为标签
        
        # 加载数据
        data = load_and_validate_data(str(temp_file))
        # 数据加载成功
        
        # 检查是否有behavior列
        if 'behavior' not in data.columns:
            # 清理临时文件
            temp_file.unlink()
            raise HTTPException(
                status_code=400, 
                detail="数据文件缺少behavior列。请确保上传的文件包含行为标签数据。"
            )
        
        # 获取所有唯一的行为类型（去除空值和'Unknown'）
        unique_behaviors = data['behavior'].dropna().unique()
        unique_behaviors = [behavior for behavior in unique_behaviors if behavior != 'Unknown' and str(behavior).strip()]
        
        # 按字母顺序排序
        unique_behaviors = sorted(unique_behaviors)
        
        # 发现行为类型
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "behaviors": unique_behaviors,
            "total_behaviors": len(unique_behaviors),
            "filename": file.filename,
            "message": f"成功提取到 {len(unique_behaviors)} 种行为标签"
        }
        
    except Exception as e:
        print(f"提取行为标签错误: {e}")
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        
        # 如果是HTTPException，直接重新抛出
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(status_code=500, detail=f"提取行为标签失败: {str(e)}")

@app.post("/api/heatmap/analyze")
async def heatmap_analysis(
    file: UploadFile = File(...),
    start_behavior: str = Form(...),
    end_behavior: str = Form(...),
    pre_behavior_time: float = Form(15.0),
    post_behavior_time: float = Form(45.0),
    min_duration: float = Form(1.0),
    sampling_rate: float = Form(4.8)
):
    """热力图分析"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 创建配置对象
        config = BehaviorHeatmapConfig()
        config.INPUT_FILE = str(temp_file)
        config.START_BEHAVIOR = start_behavior
        config.END_BEHAVIOR = end_behavior
        config.PRE_BEHAVIOR_TIME = pre_behavior_time
        config.POST_BEHAVIOR_TIME = post_behavior_time
        config.MIN_BEHAVIOR_DURATION = min_duration
        config.SAMPLING_RATE = sampling_rate
        config.OUTPUT_DIR = str(RESULTS_DIR / "heatmaps")
        config.SORTING_METHOD = 'first'
        
        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # 加载数据
        data = load_and_validate_data(str(temp_file))
        
        # 检查是否有有效的行为数据
        if 'behavior' in data.columns:
            unique_behaviors = data['behavior'].unique()
            if len(unique_behaviors) == 1 and unique_behaviors[0] == 'Unknown':
                raise HTTPException(
                    status_code=400, 
                    detail=f"数据文件缺少行为标签信息。当前文件只包含神经元活动数据，无法进行行为热力分析。请上传包含行为标签的数据文件。"
                )
        
        # 查找行为配对
        behavior_pairs = find_behavior_pairs(
            data, start_behavior, end_behavior, 
            min_duration, sampling_rate
        )
        
        if not behavior_pairs:
            available_behaviors = data['behavior'].unique() if 'behavior' in data.columns else []
            raise HTTPException(
                status_code=400, 
                detail=f"未找到从'{start_behavior}'到'{end_behavior}'的行为配对。数据中可用的行为类型: {list(available_behaviors)}"
            )
        
        # 提取所有行为序列数据
        all_sequence_data = []
        heatmap_images = []
        first_heatmap_order = None
        valid_pairs_count = 0
        
        for i, (start_begin, start_end, end_begin, end_end) in enumerate(behavior_pairs):
            # 提取行为序列数据
            sequence_data = extract_behavior_sequence_data(
                data, start_begin, end_end, pre_behavior_time, post_behavior_time, sampling_rate
            )
            
            if sequence_data is not None:
                # 标准化数据
                standardized_data = standardize_neural_data(sequence_data)
                all_sequence_data.append(standardized_data)
                valid_pairs_count += 1
                
                # 创建热力图
                fig, current_order = create_behavior_sequence_heatmap(
                    standardized_data, start_begin, end_end,
                    start_behavior, end_behavior, pre_behavior_time,
                    post_behavior_time, config, i, first_heatmap_order=first_heatmap_order
                )
                
                # 保存第一个热力图的排序顺序
                if valid_pairs_count == 1 and current_order is not None:
                    first_heatmap_order = current_order
                
                # 将图表转换为base64
                plot_base64 = save_plot_as_base64(fig)
                heatmap_images.append({
                    "title": f"行为配对 {valid_pairs_count} 热力图",
                    "url": plot_base64
                })
            else:
                # 跳过时间范围超出数据范围的行为配对
                pass
        
        # 检查是否有有效的序列数据
        if not all_sequence_data:
            raise HTTPException(
                status_code=400,
                detail=f"无法提取有效的行为序列数据。所有找到的行为配对的时间范围都超出了数据范围。请检查行为时间和预行为时间设置。"
            )
        
        # 创建平均热力图
        if len(all_sequence_data) > 1:
            avg_fig = create_average_sequence_heatmap(
                all_sequence_data, start_behavior, end_behavior,
                pre_behavior_time, post_behavior_time, config, first_heatmap_order=first_heatmap_order
            )
            avg_plot_base64 = save_plot_as_base64(avg_fig)
            heatmap_images.append({
                "title": "平均热力图",
                "url": avg_plot_base64
            })
        
        # 清理临时文件
        temp_file.unlink()
        
        # 获取神经元数量
        neuron_columns = [col for col in data.columns if col not in ['behavior']]
        
        return {
            "success": True,
            "filename": file.filename,
            "behavior_pairs_count": len(behavior_pairs),
            "neuron_count": len(neuron_columns),
            "start_behavior": start_behavior,
            "end_behavior": end_behavior,
            "status": "分析完成",
            "heatmap_images": heatmap_images
        }
        
    except Exception as e:
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        
        # 详细的错误日志记录
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"行为热力分析错误详情: {error_details}")
        
        # 如果是HTTPException，直接重新抛出以保持原始错误信息
        if isinstance(e, HTTPException):
            raise e
        
        # 对于其他异常，提供详细的错误信息
        error_message = str(e) if str(e) else f"未知错误: {type(e).__name__}"
        raise HTTPException(status_code=500, detail=f"分析失败: {error_message}")

@app.post("/api/heatmap/overall")
async def overall_heatmap_analysis(
    file: UploadFile = File(...),
    stamp_min: Optional[float] = Form(None),
    stamp_max: Optional[float] = Form(None),
    sort_method: str = Form("peak"),
    calcium_wave_threshold: float = Form(1.5),
    min_prominence: float = Form(1.0),
    min_rise_rate: float = Form(0.1),
    max_fall_rate: float = Form(0.05)
):
    """整体热力图分析"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"overall_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 整体热力图分析
        
        # 读取数据 - 支持多种文件格式
        if file.filename.endswith('.csv'):
            data = pd.read_csv(temp_file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(temp_file)
        else:
            # 尝试自动检测文件格式
            try:
                data = pd.read_csv(temp_file)
            except:
                try:
                    data = pd.read_excel(temp_file)
                except:
                    raise ValueError(f"不支持的文件格式: {file.filename}")
        # 数据加载成功
        
        # 创建配置对象
        config = OverallHeatmapConfig()
        config.STAMP_MIN = stamp_min
        config.STAMP_MAX = stamp_max
        config.SORT_METHOD = sort_method
        config.CALCIUM_WAVE_THRESHOLD = calcium_wave_threshold
        config.MIN_PROMINENCE = min_prominence
        config.MIN_RISE_RATE = min_rise_rate
        config.MAX_FALL_RATE = max_fall_rate
        
        # 生成整体热力图
        fig, info = generate_overall_heatmap(data, config)
        
        # 将图表转换为base64
        plot_base64 = save_plot_as_base64(fig)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "filename": file.filename,
            "heatmap_image": plot_base64,
            "analysis_info": info,
            "config": {
                "stamp_min": stamp_min,
                "stamp_max": stamp_max,
                "sort_method": sort_method,
                "calcium_wave_threshold": calcium_wave_threshold,
                "min_prominence": min_prominence,
                "min_rise_rate": min_rise_rate,
                "max_fall_rate": max_fall_rate
            },
            "message": "整体热力图生成完成"
        }
        
    except Exception as e:
        print(f"整体热力图分析错误: {e}")
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"整体热力图分析失败: {str(e)}")

@app.post("/api/heatmap/em-sort-labels")
async def get_em_sort_labels(file: UploadFile = File(...)):
    """获取EM排序数据中的神经元标签"""
    try:
        # 读取文件内容
        content = await file.read()
        
        # 根据文件扩展名选择处理方式
        if file.filename.endswith('.csv'):
            import pandas as pd
            import io
            
            # 读取CSV文件
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # 获取神经元列（除了时间列）
            time_columns = ['time', 'timestamp', 'stamp', 't']
            neuron_columns = [col for col in df.columns if col.lower() not in time_columns]
            
            return {
                "success": True,
                "labels": neuron_columns,
                "total_neurons": len(neuron_columns)
            }
        else:
            return {
                "success": False,
                "error": "不支持的文件格式，请上传CSV文件"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"读取文件失败: {str(e)}"
        }

@app.post("/api/heatmap/em-sort")
async def em_sort_heatmap_analysis(
    file: UploadFile = File(...),
    stamp_min: Optional[float] = Form(None),  # 不填时使用整个数据范围
    stamp_max: Optional[float] = Form(None),  # 不填时使用整个数据范围
    sort_method: str = Form("peak"),
    custom_neuron_order: Optional[str] = Form(None),
    calcium_wave_threshold: float = Form(1.5),
    min_prominence: float = Form(1.0),
    min_rise_rate: float = Form(0.1),
    max_fall_rate: float = Form(0.05),
    sampling_rate: float = Form(4.8)
):
    """EM排序热力图分析"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"em_sort_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # EM排序热力图分析
        
        # 读取数据 - 支持多种文件格式
        if file.filename.endswith('.csv'):
            data = pd.read_csv(temp_file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(temp_file)
        else:
            # 尝试自动检测文件格式
            try:
                data = pd.read_csv(temp_file)
            except:
                try:
                    data = pd.read_excel(temp_file)
                except:
                    raise ValueError(f"不支持的文件格式: {file.filename}")
        # 数据加载成功
        
        # 解析自定义神经元顺序
        custom_order = None
        if custom_neuron_order:
            try:
                # 假设传入的是逗号分隔的字符串
                custom_order = [neuron.strip() for neuron in custom_neuron_order.split(',') if neuron.strip()]
            except:
                custom_order = None
        
        # 创建配置对象
        config = EMSortHeatmapConfig(
            stamp_min=stamp_min,
            stamp_max=stamp_max,
            sort_method=sort_method,
            custom_neuron_order=custom_order,
            calcium_wave_threshold=calcium_wave_threshold,
            min_prominence=min_prominence,
            min_rise_rate=min_rise_rate,
            max_fall_rate=max_fall_rate,
            sampling_rate=sampling_rate
        )
        
        # 生成EM排序热力图
        fig, info = analyze_em_sort_heatmap(data, config)
        
        # 将图表转换为base64
        plot_base64 = save_plot_as_base64(fig)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "filename": file.filename,
            "heatmap_image": plot_base64,
            "analysis_info": info,
            "config": {
                "stamp_min": stamp_min,
                "stamp_max": stamp_max,
                "sort_method": sort_method,
                "custom_neuron_order": custom_neuron_order,
                "calcium_wave_threshold": calcium_wave_threshold,
                "min_prominence": min_prominence,
                "min_rise_rate": min_rise_rate,
                "max_fall_rate": max_fall_rate,
                "sampling_rate": sampling_rate
            },
            "message": "EM排序热力图生成完成"
        }
        
    except Exception as e:
        print(f"EM排序热力图分析错误: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"EM排序热力图分析失败: {str(e)}")

@app.post("/api/heatmap/multi-day")
async def multi_day_heatmap_analysis(
    files: List[UploadFile] = File(...),
    day_labels: str = Form(...),  # 逗号分隔的天数标签，如 "day0,day3,day6,day9"
    sort_method: str = Form("peak"),
    calcium_wave_threshold: float = Form(1.5),
    min_prominence: float = Form(1.0),
    min_rise_rate: float = Form(0.1),
    max_fall_rate: float = Form(0.05),
    create_combination: bool = Form(True),
    create_individual: bool = Form(True)
):
    """多天数据组合热力图分析"""
    try:
        # 解析天数标签
        day_labels_list = [label.strip() for label in day_labels.split(',') if label.strip()]
        
        if len(files) != len(day_labels_list):
            raise HTTPException(
                status_code=400, 
                detail=f"文件数量({len(files)})与天数标签数量({len(day_labels_list)})不匹配"
            )
        
        # 保存上传的文件并读取数据
        data_dict = {}
        temp_files = []
        
        for i, (file, day_label) in enumerate(zip(files, day_labels_list)):
            temp_file = TEMP_DIR / f"multiday_{day_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            temp_files.append(temp_file)
            
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 读取数据 - 支持多种文件格式
            if file.filename.endswith('.csv'):
                data = pd.read_csv(temp_file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(temp_file)
            else:
                # 尝试自动检测文件格式
                try:
                    data = pd.read_csv(temp_file)
                except:
                    try:
                        data = pd.read_excel(temp_file)
                    except:
                        raise ValueError(f"不支持的文件格式: {file.filename}")
            data_dict[day_label] = data
            # 数据加载成功
        
        # 创建配置对象
        config = MultiDayHeatmapConfig(
            sort_method=sort_method,
            calcium_wave_threshold=calcium_wave_threshold,
            min_prominence=min_prominence,
            min_rise_rate=min_rise_rate,
            max_fall_rate=max_fall_rate
        )
        
        # 执行多天热力图分析
        results = analyze_multiday_heatmap(
            data_dict, 
            config, 
            correspondence_table=None,  # 暂时不支持对应表
            create_combination=create_combination,
            create_individual=create_individual
        )
        
        # 转换图形为base64
        response_data = {
            "success": True,
            "filenames": [file.filename for file in files],
            "day_labels": day_labels_list,
            "analysis_info": results['analysis_info'],
            "config": {
                "sort_method": sort_method,
                "calcium_wave_threshold": calcium_wave_threshold,
                "min_prominence": min_prominence,
                "min_rise_rate": min_rise_rate,
                "max_fall_rate": max_fall_rate,
                "create_combination": create_combination,
                "create_individual": create_individual
            }
        }
        
        # 添加组合热力图
        if results['combination_heatmap']:
            combo_base64 = save_plot_as_base64(results['combination_heatmap']['figure'])
            response_data['combination_heatmap'] = {
                "image": combo_base64,
                "info": results['combination_heatmap']['info']
            }
        
        # 添加单独热力图
        individual_heatmaps = []
        for day, heatmap_data in results['individual_heatmaps'].items():
            individual_base64 = save_plot_as_base64(heatmap_data['figure'])
            individual_heatmaps.append({
                "day": day,
                "image": individual_base64,
                "info": heatmap_data['info']
            })
        
        response_data['individual_heatmaps'] = individual_heatmaps
        response_data['message'] = f"多天热力图分析完成，处理了{len(day_labels_list)}天的数据"
        
        # 清理临时文件
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        return response_data
        
    except Exception as e:
        print(f"多天热力图分析错误: {e}")
        # 清理临时文件
        if 'temp_files' in locals():
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
        
        # 如果是HTTPException，直接重新抛出
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(status_code=500, detail=f"多天热力图分析失败: {str(e)}")

@app.post("/api/trace/analyze")
async def trace_analysis(
    file: UploadFile = File(...),
    stamp_min: Optional[float] = Form(None),
    stamp_max: Optional[float] = Form(None),
    sort_method: str = Form("peak"),
    custom_neuron_order: Optional[str] = Form(None),
    trace_offset: float = Form(60.0),
    scaling_factor: float = Form(80.0),
    max_neurons: int = Form(60),
    trace_alpha: float = Form(0.8),
    line_width: float = Form(2.0),
    sampling_rate: float = Form(4.8),
    calcium_wave_threshold: float = Form(1.5),
    min_prominence: float = Form(1.0),
    min_rise_rate: float = Form(0.1),
    max_fall_rate: float = Form(0.05)
):
    """
    执行Trace图分析
    
    参数:
    - file: 数据文件
    - stamp_min: 最小时间戳
    - stamp_max: 最大时间戳
    - sort_method: 排序方式 ('original', 'peak', 'calcium_wave', 'custom')
    - custom_neuron_order: 自定义神经元顺序（JSON格式）
    - trace_offset: 神经元间垂直偏移
    - scaling_factor: 信号振幅缩放因子
    - max_neurons: 最大显示神经元数量
    - trace_alpha: 线条透明度
    - line_width: 线条宽度
    - sampling_rate: 采样率
    - calcium_wave_threshold: 钙波检测阈值
    - min_prominence: 最小峰值突出度
    - min_rise_rate: 最小上升速率
    - max_fall_rate: 最大下降速率
    """
    try:
        # 保存上传的文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        # 加载数据
        data = load_trace_data(temp_file.name)
        
        # 解析自定义神经元顺序
        custom_order = []
        if custom_neuron_order:
            try:
                custom_order = json.loads(custom_neuron_order)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="自定义神经元顺序格式错误，应为JSON数组格式")
        
        # 创建配置对象
        config = TraceConfig()
        config.stamp_min = stamp_min
        config.stamp_max = stamp_max
        config.sort_method = sort_method
        config.custom_neuron_order = custom_order
        config.trace_offset = trace_offset
        config.scaling_factor = scaling_factor
        config.max_neurons = max_neurons
        config.trace_alpha = trace_alpha
        config.line_width = line_width
        config.sampling_rate = sampling_rate
        config.calcium_wave_threshold = calcium_wave_threshold
        config.min_prominence = min_prominence
        config.min_rise_rate = min_rise_rate
        config.max_fall_rate = max_fall_rate
        
        # 分离行为数据
        behavior_data = None
        if 'behavior' in data.columns:
            behavior_data = data['behavior']
            trace_data = data.drop(columns=['behavior'])
        else:
            trace_data = data
        
        # 将时间戳设置为索引
        if 'stamp' in trace_data.columns:
            trace_data = trace_data.set_index('stamp')
            if behavior_data is not None:
                # behavior_data已经是Series，需要重新创建索引
                behavior_data = pd.Series(behavior_data.values, index=data['stamp'])
        
        # 生成Trace图
        image_base64, info = generate_trace_plot(trace_data, behavior_data, config)
        
        # 清理临时文件
        os.unlink(temp_file.name)
        
        # 返回结果
        result = {
            "success": True,
            "image": image_base64,
            "info": info,
            "request_params": {
                "filename": file.filename,
                "stamp_min": stamp_min,
                "stamp_max": stamp_max,
                "sort_method": sort_method,
                "custom_neuron_order": custom_order,
                "trace_offset": trace_offset,
                "scaling_factor": scaling_factor,
                "max_neurons": max_neurons,
                "trace_alpha": trace_alpha,
                "line_width": line_width,
                "sampling_rate": sampling_rate,
                "calcium_wave_threshold": calcium_wave_threshold,
                "min_prominence": min_prominence,
                "min_rise_rate": min_rise_rate,
                "max_fall_rate": max_fall_rate
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Trace图分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trace图分析失败: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载结果文件"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# ============================================================================
# 神经元分析API端点
# ============================================================================

@app.post("/api/neuron/effect-size")
async def effect_size_analysis(
    file: UploadFile = File(...),
    behavior_column: Optional[str] = Form(None),
    threshold: float = Form(0.5)
):
    """效应量分析API"""
    try:
        # 保存上传的文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        # 读取数据
        data = pd.read_excel(temp_file.name)
        
        # 确保数值列的数据类型正确
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 执行效应量分析
        result = analyze_effect_sizes(data, behavior_column)
        
        # 清理临时文件
        os.unlink(temp_file.name)
        
        # 转换效应量数据结构以匹配前端期望
        effect_sizes = result.get('effect_sizes', {})
        behavior_labels = result.get('behavior_labels', [])
        
        # 将效应量数据从 {behavior: [neuron_effects]} 转换为 {neuron_id: {behavior: effect}}
        formatted_effect_sizes = {}
        if effect_sizes and behavior_labels:
            for behavior, effects_array in effect_sizes.items():
                for neuron_idx, effect_value in enumerate(effects_array):
                    neuron_id = f"Neuron_{neuron_idx + 1}"
                    if neuron_id not in formatted_effect_sizes:
                        formatted_effect_sizes[neuron_id] = {}
                    formatted_effect_sizes[neuron_id][behavior] = float(effect_value)
        
        # 重新组织统计数据结构，确保所有数据都是Python原生类型
        data_summary = result.get('data_summary', {})
        
        # 转换behavior_counts中的numpy类型
        behavior_counts = data_summary.get('behavior_counts', {})
        if behavior_counts:
            behavior_counts = {str(k): int(v) if hasattr(v, 'item') else v for k, v in behavior_counts.items()}
        
        statistics = {
            'total_neurons': int(data_summary.get('total_neurons', 0)),
            'total_behaviors': int(len(behavior_labels)),
            'total_samples': int(data_summary.get('total_samples', 0)),
            'key_neurons_found': int(sum(len(neurons.get('neuron_ids', [])) for neurons in result.get('top_neurons', {}).values())),
            'analysis_timestamp': datetime.now().isoformat(),
            'behavior_counts': behavior_counts
        }
        
        # 转换key_neurons中的numpy类型
        key_neurons = result.get('top_neurons', {})
        formatted_key_neurons = {}
        for behavior, neuron_info in key_neurons.items():
            formatted_key_neurons[behavior] = {
                'neuron_ids': [str(neuron_id) for neuron_id in neuron_info.get('neuron_ids', [])],
                'effect_sizes': [float(effect) for effect in neuron_info.get('effect_sizes', [])],
                'abs_effect_sizes': [float(effect) for effect in neuron_info.get('abs_effect_sizes', [])]
            }
        
        # 转换nan_info中的numpy类型
        nan_info = result.get('nan_info', {})
        formatted_nan_info = {}
        for key, value in nan_info.items():
            if hasattr(value, 'item'):  # numpy scalar
                formatted_nan_info[key] = value.item()
            elif isinstance(value, (list, tuple)) and len(value) > 0 and hasattr(value[0], 'item'):
                formatted_nan_info[key] = [v.item() if hasattr(v, 'item') else v for v in value]
            else:
                formatted_nan_info[key] = value
        
        # 重新组织数据结构以匹配前端期望
        formatted_result = {
            "effect_sizes": formatted_effect_sizes,
            "behavior_labels": [str(label) for label in behavior_labels],
            "key_neurons": formatted_key_neurons,
            "statistics": statistics,
            "nan_info": formatted_nan_info,
            "processed_data": {}  # 不返回处理后的数据，避免numpy类型问题
        }
        
        return {
            "success": True,
            "result": formatted_result,
            "request_params": {
                "filename": file.filename,
                "behavior_column": behavior_column,
                "threshold": threshold
            }
        }
        
    except Exception as e:
        print(f"效应量分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"效应量分析失败: {str(e)}")

@app.post("/api/neuron/position")
async def position_analysis(
    positions_data: str = Form(...)
):
    """位置分析API"""
    try:
        # 解析位置数据
        positions_dict = json.loads(positions_data)
        
        # 处理位置数据
        result = process_position_data_v2(positions_dict)
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        print(f"位置分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"位置分析失败: {str(e)}")

@app.post("/api/neuron/principal-analysis")
async def principal_neuron_analysis(
    file: UploadFile = File(...),
    behavior_column: Optional[str] = Form(None),
    positions_data: Optional[str] = Form(None),
    threshold: float = Form(0.5)
):
    """主神经元分析API"""
    try:
        # 保存上传的文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        # 读取数据
        data = pd.read_excel(temp_file.name)
        
        # 确保数值列的数据类型正确
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 解析位置数据（如果有）
        positions_dict = None
        if positions_data:
            positions_dict = json.loads(positions_data)
        
        # 执行主神经元分析
        result = analyze_principal_neurons(data, behavior_column, positions_dict, threshold)
        
        # 清理临时文件
        os.unlink(temp_file.name)
        
        return {
            "success": True,
            "result": result,
            "request_params": {
                "filename": file.filename,
                "behavior_column": behavior_column,
                "threshold": threshold,
                "has_positions": positions_dict is not None
            }
        }
        
    except Exception as e:
        print(f"主神经元分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"主神经元分析失败: {str(e)}")

@app.post("/api/neuron/comprehensive-analysis")
async def comprehensive_neuron_analysis(
    effect_size_data: str = Form(...),
    position_data: Optional[str] = Form(None),
    analysis_type: str = Form("effect-position"),
    visualization_options: Optional[str] = Form(None),
    selected_behavior: Optional[str] = Form(None),
    neuron_type: Optional[str] = Form("key"),
    display_options: Optional[str] = Form(None)
):
    """综合神经元分析API - 基于已有的效应量分析结果和位置数据"""
    try:
        # 解析效应量分析结果
        effect_size_result = json.loads(effect_size_data)
        
        # 解析位置数据（如果有）
        position_data_dict = None
        if position_data:
            position_data_dict = json.loads(position_data)
        
        # 解析可视化选项
        viz_options = []
        if visualization_options:
            viz_options = json.loads(visualization_options)
        
        # 解析显示选项
        display_opts = []
        if display_options:
            display_opts = json.loads(display_options)
        
        # 基于分析类型执行不同的分析
        comprehensive_result = {
            "analysis_type": analysis_type,
            "effect_size_analysis": effect_size_result,
            "position_analysis": position_data_dict,
            "visualization_options": viz_options
        }
        
        # 根据分析类型生成不同的可视化结果
        if analysis_type == "effect-position" and position_data_dict:
            # 效应量-位置关联分析
            comprehensive_result["effect_position_plot"] = generate_effect_position_plot(
                effect_size_result, position_data_dict
            )
        elif analysis_type == "spatial-clustering" and position_data_dict:
            # 空间聚类分析
            comprehensive_result["spatial_clustering_plot"] = generate_spatial_clustering_plot(
                effect_size_result, position_data_dict
            )
        elif analysis_type == "behavior-position-heatmap" and position_data_dict:
            # 行为-位置热力图
            comprehensive_result["behavior_position_heatmap"] = generate_behavior_position_heatmap(
                effect_size_result, position_data_dict
            )
        elif analysis_type == "comprehensive" and position_data_dict:
            # 综合可视化
            comprehensive_result["comprehensive_visualization"] = generate_comprehensive_visualization(
                effect_size_result, position_data_dict, viz_options
            )
        elif analysis_type == "single-behavior-spatial" and position_data_dict and selected_behavior:
            # 单行为空间分析
            comprehensive_result["single_behavior_spatial_plot"] = generate_single_behavior_spatial_plot(
                effect_size_result, position_data_dict, selected_behavior, neuron_type, display_opts
            )
        
        # 添加统计信息
        comprehensive_result["statistics"] = {
            "total_neurons": effect_size_result.get("statistics", {}).get("total_neurons", 0),
            "total_behaviors": effect_size_result.get("statistics", {}).get("total_behaviors", 0),
            "total_positions": len(position_data_dict.get("positions", {})) if position_data_dict else 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "result": comprehensive_result,
            "request_params": {
                "analysis_type": analysis_type,
                "has_positions": position_data_dict is not None,
                "visualization_options": viz_options
            }
        }
        
    except Exception as e:
        print(f"综合神经元分析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"综合神经元分析失败: {str(e)}")


def generate_effect_position_plot(effect_size_result, position_data):
    """生成效应量-位置关联图"""
    try:
        # 配置中文字体
        configure_chinese_font()
        
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取效应量数据
        effect_sizes = effect_size_result.get('effect_sizes', {})
        positions = position_data.get('positions', {})
        
        # 提取数据用于绘图
        neuron_ids = []
        max_effects = []
        x_coords = []
        y_coords = []
        
        for neuron_id, effects in effect_sizes.items():
            if neuron_id in positions:
                neuron_ids.append(neuron_id)
                # 计算该神经元的最大效应量
                max_effect = max([abs(effect) for effect in effects.values()])
                max_effects.append(max_effect)
                x_coords.append(positions[neuron_id]['x'])
                y_coords.append(positions[neuron_id]['y'])
        
        # 绘制散点图
        scatter = ax.scatter(x_coords, y_coords, c=max_effects, cmap='viridis', s=100, alpha=0.7)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='最大效应量')
        
        # 设置标题和标签
        ax.set_title('神经元效应量-位置分布图')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
        
    except Exception as e:
        print(f"生成效应量-位置图错误: {e}")
        return None


def generate_spatial_clustering_plot(effect_size_result, position_data):
    """生成空间聚类图"""
    try:
        # 配置中文字体
        configure_chinese_font()
        
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        from sklearn.cluster import KMeans
        import numpy as np
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取位置数据
        positions = position_data.get('positions', {})
        
        # 提取坐标
        coords = []
        neuron_ids = []
        for neuron_id, pos in positions.items():
            coords.append([pos['x'], pos['y']])
            neuron_ids.append(neuron_id)
        
        if len(coords) < 3:
            ax.text(0.5, 0.5, '位置数据不足，无法进行聚类分析', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # 执行K-means聚类
            coords_array = np.array(coords)
            n_clusters = min(3, len(coords) // 2)  # 最多3个聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords_array)
            
            # 绘制聚类结果
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i in range(n_clusters):
                mask = cluster_labels == i
                ax.scatter(coords_array[mask, 0], coords_array[mask, 1], 
                          c=colors[i % len(colors)], label=f'聚类 {i+1}', s=100, alpha=0.7)
            
            # 绘制聚类中心
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      c='black', marker='x', s=200, linewidths=3, label='聚类中心')
            
            ax.legend()
        
        # 设置标题和标签
        ax.set_title('神经元空间聚类分析')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
        
    except Exception as e:
        print(f"生成空间聚类图错误: {e}")
        return None


def generate_behavior_position_heatmap(effect_size_result, position_data):
    """生成行为-位置热力图"""
    try:
        # 配置中文字体
        configure_chinese_font()
        
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        import numpy as np
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 获取数据
        effect_sizes = effect_size_result.get('effect_sizes', {})
        behavior_labels = effect_size_result.get('behavior_labels', [])
        positions = position_data.get('positions', {})
        
        # 创建热力图数据
        heatmap_data = []
        for behavior in behavior_labels:
            behavior_effects = []
            for neuron_id in positions.keys():
                if neuron_id in effect_sizes and behavior in effect_sizes[neuron_id]:
                    behavior_effects.append(effect_sizes[neuron_id][behavior])
                else:
                    behavior_effects.append(0)
            heatmap_data.append(behavior_effects)
        
        if heatmap_data:
            # 绘制热力图
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
            
            # 设置标签
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(list(positions.keys()), rotation=45)
            ax.set_yticks(range(len(behavior_labels)))
            ax.set_yticklabels(behavior_labels)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='效应量')
            
            # 设置标题
            ax.set_title('行为-神经元效应量热力图')
            ax.set_xlabel('神经元ID')
            ax.set_ylabel('行为类型')
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
        
    except Exception as e:
        print(f"生成行为-位置热力图错误: {e}")
        return None


def generate_comprehensive_visualization(effect_size_result, position_data, viz_options):
    """生成综合可视化"""
    try:
        # 配置中文字体
        configure_chinese_font()
        
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 效应量分布直方图
        effect_sizes = effect_size_result.get('effect_sizes', {})
        all_effects = []
        for neuron_effects in effect_sizes.values():
            all_effects.extend([abs(effect) for effect in neuron_effects.values()])
        
        ax1.hist(all_effects, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('效应量分布直方图')
        ax1.set_xlabel('效应量绝对值')
        ax1.set_ylabel('频次')
        
        # 2. 神经元位置分布
        positions = position_data.get('positions', {})
        x_coords = [pos['x'] for pos in positions.values()]
        y_coords = [pos['y'] for pos in positions.values()]
        
        ax2.scatter(x_coords, y_coords, alpha=0.7, s=50)
        ax2.set_title('神经元位置分布')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        
        # 3. 行为类型统计
        behavior_labels = effect_size_result.get('behavior_labels', [])
        behavior_counts = effect_size_result.get('statistics', {}).get('behavior_counts', {})
        
        if behavior_counts:
            behaviors = list(behavior_counts.keys())
            counts = list(behavior_counts.values())
            ax3.bar(behaviors, counts, alpha=0.7, color='lightcoral')
            ax3.set_title('行为类型统计')
            ax3.set_xlabel('行为类型')
            ax3.set_ylabel('样本数量')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 关键神经元效应量
        key_neurons = effect_size_result.get('key_neurons', {})
        if key_neurons:
            behaviors = list(key_neurons.keys())
            neuron_counts = [len(info.get('neuron_ids', [])) for info in key_neurons.values()]
            ax4.bar(behaviors, neuron_counts, alpha=0.7, color='lightgreen')
            ax4.set_title('各行为关键神经元数量')
            ax4.set_xlabel('行为类型')
            ax4.set_ylabel('关键神经元数量')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
        
    except Exception as e:
        print(f"生成综合可视化错误: {e}")
        return None


def generate_single_behavior_spatial_plot(effect_size_result, position_data, selected_behavior, neuron_type, display_options):
    """生成单行为空间分析图"""
    try:
        # 配置中文字体
        configure_chinese_font()
        
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        import numpy as np
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取效应量数据和位置数据
        effect_sizes = effect_size_result.get('effect_sizes', {})
        positions = position_data.get('positions', {})
        
        # 准备数据
        neuron_data = []
        effect_values = []
        x_coords = []
        y_coords = []
        neuron_ids = []
        
        # 根据神经元类型筛选数据
        if neuron_type == 'key':
            # 只显示关键神经元
            key_neurons = effect_size_result.get('key_neurons', {}).get(selected_behavior, {})
            key_neuron_ids = key_neurons.get('neuron_ids', [])
            
            for neuron_id in key_neuron_ids:
                if neuron_id in positions and neuron_id in effect_sizes:
                    if selected_behavior in effect_sizes[neuron_id]:
                        neuron_data.append({
                            'id': neuron_id,
                            'x': positions[neuron_id]['x'],
                            'y': positions[neuron_id]['y'],
                            'effect': effect_sizes[neuron_id][selected_behavior]
                        })
        else:
            # 显示所有神经元
            for neuron_id, pos in positions.items():
                if neuron_id in effect_sizes and selected_behavior in effect_sizes[neuron_id]:
                    neuron_data.append({
                        'id': neuron_id,
                        'x': pos['x'],
                        'y': pos['y'],
                        'effect': effect_sizes[neuron_id][selected_behavior]
                    })
        
        if not neuron_data:
            ax.text(0.5, 0.5, f'没有找到行为 "{selected_behavior}" 的数据', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(f'行为 "{selected_behavior}" 空间分析 - 无数据')
        else:
            # 提取坐标和效应量
            x_coords = [item['x'] for item in neuron_data]
            y_coords = [item['y'] for item in neuron_data]
            effect_values = [abs(item['effect']) for item in neuron_data]  # 使用绝对值
            neuron_ids = [item['id'] for item in neuron_data]
            
            # 绘制散点图，颜色表示效应量大小
            scatter = ax.scatter(x_coords, y_coords, c=effect_values, 
                               cmap='viridis', s=100, alpha=0.8, 
                               edgecolors='white', linewidth=0.5)
            
            # 添加颜色条
            if 'show-colorbar' in display_options:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('效应量绝对值', fontsize=12)
            
            # 显示神经元ID
            if 'show-neuron-ids' in display_options:
                for i, neuron_id in enumerate(neuron_ids):
                    ax.annotate(neuron_id, (x_coords[i], y_coords[i]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.8, color='white',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
            
            # 显示效应量数值
            if 'show-effect-values' in display_options:
                for i, (neuron_id, effect) in enumerate(zip(neuron_ids, effect_values)):
                    ax.annotate(f'{effect:.3f}', (x_coords[i], y_coords[i]),
                              xytext=(0, -15), textcoords='offset points',
                              fontsize=7, alpha=0.8, ha='center',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # 设置标题和标签
        neuron_type_text = "关键神经元" if neuron_type == 'key' else "所有神经元"
        ax.set_title(f'行为 "{selected_behavior}" 空间分析 - {neuron_type_text}\n'
                    f'({len(neuron_data)} 个神经元)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # 添加统计信息
        if neuron_data:
            max_effect = max(effect_values)
            min_effect = min(effect_values)
            mean_effect = np.mean(effect_values)
            
            stats_text = f'效应量范围: {min_effect:.3f} - {max_effect:.3f}\n平均效应量: {mean_effect:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
        
    except Exception as e:
        print(f"生成单行为空间分析图错误: {e}")
        return None

# ============================================================================
# 三步走工作流API端点
# ============================================================================

@app.post("/api/workflow/step1-effect-size")
async def step1_effect_size_analysis(
    file: UploadFile = File(...),
    behavior_column: Optional[str] = Form(None)
):
    """第一步：效应量分析 - 计算神经元与行为的效应量"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"step1_effect_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 读取数据
        if file.filename.endswith('.csv'):
            data = pd.read_csv(temp_file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(temp_file)
        else:
            try:
                data = pd.read_csv(temp_file)
            except:
                try:
                    data = pd.read_excel(temp_file)
                except:
                    raise ValueError(f"不支持的文件格式: {file.filename}")
        
        # 执行效应量分析
        result = analyze_effect_sizes(data, behavior_column)
        
        # 保存效应量结果到临时文件，供后续步骤使用
        effect_size_file = TEMP_DIR / f"effect_sizes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 创建效应量DataFrame并保存
        effect_df = pd.DataFrame(result['effect_sizes'])
        effect_df.index = [f"Neuron_{i+1}" for i in range(len(effect_df))]
        effect_df.index.name = "Neuron_ID"
        effect_df.to_csv(effect_size_file)
        
        # 清理原始临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "step": 1,
            "step_name": "效应量分析",
            "filename": file.filename,
            "analysis_result": result,
            "effect_size_file": str(effect_size_file),
            "next_step": "位置标记",
            "message": "第一步完成：效应量分析完成，可以进行位置标记"
        }
        
    except Exception as e:
        print(f"第一步效应量分析错误: {e}")
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"效应量分析失败: {str(e)}")


@app.post("/api/workflow/step2-position-marking")
async def step2_position_marking(
    image_file: UploadFile = File(...),
    start_number: int = Form(1)
):
    """第二步：位置标记 - 在神经元图像上标记位置"""
    try:
        # 保存上传的图像文件
        temp_image = TEMP_DIR / f"step2_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.filename}"
        with open(temp_image, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        
        # 创建位置标记器
        marker = PositionMarker()
        marker.current_number = start_number
        
        return {
            "success": True,
            "step": 2,
            "step_name": "位置标记",
            "image_file": str(temp_image),
            "start_number": start_number,
            "next_step": "神经元分析",
            "message": "第二步完成：图像已上传，请在前端进行交互式位置标记",
            "instructions": [
                "1. 在图像上点击左键添加标记点",
                "2. 右键点击撤销上一个点",
                "3. 拖拽调整已标记点的位置",
                "4. 完成后点击保存按钮"
            ]
        }
        
    except Exception as e:
        print(f"第二步位置标记错误: {e}")
        if 'temp_image' in locals() and temp_image.exists():
            temp_image.unlink()
        raise HTTPException(status_code=500, detail=f"位置标记失败: {str(e)}")


@app.post("/api/workflow/step2-save-positions")
async def step2_save_positions(
    points_data: str = Form(...)  # JSON字符串格式的标记点数据
):
    """保存位置标记数据"""
    try:
        # 解析标记点数据
        points = json.loads(points_data)
        
        # 处理位置数据
        result = process_position_data_v2(points)
        
        # 保存到临时文件
        position_file = TEMP_DIR / f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 创建位置标记器并导出数据
        marker = PositionMarker()
        for point in points:
            marker.add_point(
                x=point['x'],
                y=point['y'],
                neuron_id=point.get('neuron_id')
            )
        
        marker.export_to_csv(str(position_file))
        
        return {
            "success": True,
            "position_file": str(position_file),
            "total_points": result['total_points'],
            "statistics": result['statistics'],
            "message": f"成功保存 {result['total_points']} 个标记点"
        }
        
    except Exception as e:
        print(f"保存位置标记错误: {e}")
        raise HTTPException(status_code=500, detail=f"保存位置标记失败: {str(e)}")


@app.post("/api/workflow/step2-validate-positions")
async def step2_validate_positions(
    position_file: UploadFile = File(...)
):
    """验证位置标记数据"""
    try:
        # 保存上传的位置文件
        temp_file = TEMP_DIR / f"validate_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{position_file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(position_file.file, buffer)
        
        # 读取位置数据
        position_data = pd.read_csv(temp_file)
        
        # 验证数据
        validation_result = validate_position_data(position_data)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "validation_result": validation_result,
            "message": "位置数据验证完成"
        }
        
    except Exception as e:
        print(f"验证位置数据错误: {e}")
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"验证位置数据失败: {str(e)}")


@app.post("/api/workflow/step3-neuron-analysis")
async def step3_neuron_analysis(
    effect_size_file: str = Form(...),
    position_file: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    """第三步：神经元分析 - 结合效应量和位置数据进行可视化分析"""
    try:
        # 读取效应量数据
        if not os.path.exists(effect_size_file):
            raise ValueError(f"效应量文件不存在: {effect_size_file}")
        
        effect_data = pd.read_csv(effect_size_file, index_col=0)
        
        # 保存位置数据文件
        temp_position = TEMP_DIR / f"step3_position_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{position_file.filename}"
        with open(temp_position, "wb") as buffer:
            shutil.copyfileobj(position_file.file, buffer)
        
        # 读取位置数据
        position_data = pd.read_csv(temp_position)
        
        # 执行神经元可视化分析
        visualization_result = analyze_neuron_visualization(effect_data, position_data, threshold)
        
        # 将图形转换为base64编码
        visualization_images = {}
        for name, fig in visualization_result['figures'].items():
            # 将图形转换为base64
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            visualization_images[name] = f"data:image/png;base64,{image_base64}"
            plt.close(fig)  # 关闭图形以释放内存
        
        # 清理临时文件
        temp_position.unlink()
        
        return {
            "success": True,
            "step": 3,
            "step_name": "神经元分析",
            "analysis_result": visualization_result['analysis_result'],
            "visualization_images": visualization_images,
            "threshold": threshold,
            "message": "第三步完成：神经元分析完成，生成了可视化结果"
        }
        
    except Exception as e:
        print(f"第三步神经元分析错误: {e}")
        if 'temp_position' in locals() and temp_position.exists():
            temp_position.unlink()
        raise HTTPException(status_code=500, detail=f"神经元分析失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        limit_max_requests=2000,
        limit_concurrency=1000,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=5,
        h11_max_incomplete_event_size=2097152  # 2MB
    )