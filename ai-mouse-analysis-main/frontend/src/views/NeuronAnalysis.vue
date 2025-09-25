<template>
  <div class="neuron-analysis">
    <el-container>
      <!-- 页面标题 -->
      <el-header class="page-header">
        <h1 class="page-title">
          <el-icon><DataAnalysis /></el-icon>
          神经元分析
        </h1>
        <p class="page-description">
          综合效应量分析、位置标记和主神经元识别功能
        </p>
      </el-header>

      <el-container>
        <!-- 左侧参数面板 -->
        <el-aside :width="sidebarWidth" class="parameter-panel">
          <el-card class="parameter-card">
            <template #header>
              <div class="card-header">
                <el-icon><Setting /></el-icon>
                <span>分析参数</span>
              </div>
            </template>

            <!-- 文件上传 -->
            <el-form :model="form" label-width="120px" class="analysis-form">
              <el-form-item label="数据文件" required>
                <el-upload
                  ref="uploadRef"
                  :auto-upload="false"
                  :on-change="handleFileChange"
                  :before-remove="handleFileRemove"
                  :limit="1"
                  accept=".xlsx,.xls,.csv"
                  drag
                >
                  <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                  <div class="el-upload__text">
                    将文件拖到此处，或<em>点击上传</em>
                  </div>
                  <template #tip>
                    <div class="el-upload__tip">
                      支持Excel和CSV格式，包含神经元活动数据和行为标签
                    </div>
                  </template>
                </el-upload>
              </el-form-item>

              <!-- 分析类型选择 -->
              <el-form-item label="分析类型">
                <el-radio-group v-model="form.analysisType" @change="handleAnalysisTypeChange">
                  <el-radio value="effect-size">效应量分析</el-radio>
                  <el-radio value="principal">主神经元分析</el-radio>
                  <el-radio value="comprehensive">综合分析</el-radio>
                </el-radio-group>
              </el-form-item>

              <!-- 行为列设置 -->
              <el-form-item label="行为列名">
                <el-input
                  v-model="form.behaviorColumn"
                  placeholder="留空则使用最后一列"
                  clearable
                />
              </el-form-item>

              <!-- 效应量阈值 -->
              <el-form-item label="效应量阈值">
                <el-slider
                  v-model="form.threshold"
                  :min="0.1"
                  :max="2.0"
                  :step="0.1"
                  show-input
                  :format-tooltip="formatThresholdTooltip"
                />
              </el-form-item>

              <!-- 位置标记工具 -->
              <el-form-item v-if="form.analysisType !== 'effect-size'" label="位置标记">
                <el-switch
                  v-model="form.hasPositionData"
                  active-text="使用位置标记"
                  inactive-text="不使用位置标记"
                />
              </el-form-item>

              <!-- 操作按钮 -->
              <el-form-item>
                <el-button
                  type="primary"
                  :loading="loading"
                  :disabled="!form.file"
                  @click="startAnalysis"
                  class="analysis-button"
                >
                  <el-icon><DataAnalysis /></el-icon>
                  {{ getAnalysisButtonText() }}
                </el-button>
                <el-button @click="resetForm" :disabled="loading">
                  <el-icon><Refresh /></el-icon>
                  重置
                </el-button>
              </el-form-item>
            </el-form>
          </el-card>
        </el-aside>

        <!-- 右侧结果面板 -->
        <el-main class="results-panel">
          <el-card v-if="!analysisResult" class="empty-result-card">
            <el-empty description="请上传数据文件并开始分析" />
          </el-card>

          <div v-else class="analysis-results">
            <!-- 位置标记工具 -->
            <el-card v-if="form.hasPositionData && form.analysisType !== 'effect-size'" class="result-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Location /></el-icon>
                  <span>神经元位置标记</span>
                </div>
              </template>
              <PositionMarker 
                @positions-updated="handlePositionsUpdated"
                ref="positionMarkerRef"
              />
            </el-card>

            <!-- 分析摘要 -->
            <el-card class="summary-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Document /></el-icon>
                  <span>分析摘要</span>
                </div>
              </template>
              <div class="summary-content">
                <el-row :gutter="20">
                  <el-col :span="6" v-for="(value, key) in getSummaryData()" :key="key">
                    <div class="summary-item">
                      <div class="summary-label">{{ getSummaryLabel(key) }}</div>
                      <div class="summary-value">{{ value }}</div>
                    </div>
                  </el-col>
                </el-row>
              </div>
            </el-card>

            <!-- 可视化结果 -->
            <el-card class="visualization-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Picture /></el-icon>
                  <span>可视化结果</span>
                </div>
              </template>
              <div class="visualization-content">
                <el-tabs v-model="activeTab" type="card">
                  <!-- 效应量分析结果 -->
                  <el-tab-pane v-if="analysisResult.effect_size_analysis" label="效应量分析" name="effect-size">
                    <div class="image-container">
                      <img 
                        v-if="analysisResult.effect_size_analysis.histogram_image"
                        :src="analysisResult.effect_size_analysis.histogram_image" 
                        alt="效应量分布直方图"
                        class="result-image"
                      />
                      <img 
                        v-if="analysisResult.effect_size_analysis.heatmap_image"
                        :src="analysisResult.effect_size_analysis.heatmap_image" 
                        alt="效应量热力图"
                        class="result-image"
                      />
                    </div>
                  </el-tab-pane>

                  <!-- 主神经元分析结果 -->
                  <el-tab-pane v-if="analysisResult.principal_neuron_analysis" label="主神经元分析" name="principal">
                    <div class="image-container">
                      <img 
                        v-if="analysisResult.principal_neuron_analysis.visualizations?.network_analysis"
                        :src="analysisResult.principal_neuron_analysis.visualizations.network_analysis" 
                        alt="网络分析图"
                        class="result-image"
                      />
                      <img 
                        v-if="analysisResult.principal_neuron_analysis.visualizations?.comprehensive_analysis"
                        :src="analysisResult.principal_neuron_analysis.visualizations.comprehensive_analysis" 
                        alt="综合分析图"
                        class="result-image"
                      />
                    </div>
                  </el-tab-pane>

                  <!-- 位置分析结果 -->
                  <el-tab-pane v-if="analysisResult.position_analysis" label="位置分析" name="position">
                    <div class="image-container">
                      <img 
                        v-if="analysisResult.position_analysis.position_plot"
                        :src="analysisResult.position_analysis.position_plot" 
                        alt="位置分布图"
                        class="result-image"
                      />
                      <img 
                        v-if="analysisResult.position_analysis.behavior_overlay_plot"
                        :src="analysisResult.position_analysis.behavior_overlay_plot" 
                        alt="行为叠加图"
                        class="result-image"
                      />
                    </div>
                  </el-tab-pane>
                </el-tabs>
              </div>
            </el-card>

            <!-- 详细数据 -->
            <el-card class="data-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Grid /></el-icon>
                  <span>详细数据</span>
                </div>
              </template>
              <div class="data-content">
                <el-tabs v-model="activeDataTab" type="card">
                  <!-- 关键神经元列表 -->
                  <el-tab-pane v-if="analysisResult.principal_neuron_analysis?.key_neurons" label="关键神经元" name="key-neurons">
                    <div v-for="(neurons, behavior) in analysisResult.principal_neuron_analysis.key_neurons" :key="behavior">
                      <h4>{{ behavior }} ({{ neurons.length }} 个神经元)</h4>
                      <el-tag 
                        v-for="neuron in neurons" 
                        :key="neuron" 
                        class="neuron-tag"
                        type="success"
                      >
                        {{ neuron }}
                      </el-tag>
                    </div>
                  </el-tab-pane>

                  <!-- 效应量数据 -->
                  <el-tab-pane v-if="analysisResult.effect_size_analysis?.effect_sizes" label="效应量数据" name="effect-sizes">
                    <el-table :data="formatEffectSizesData()" style="width: 100%" max-height="400">
                      <el-table-column prop="neuron" label="神经元" width="120" />
                      <el-table-column 
                        v-for="behavior in getBehaviors()" 
                        :key="behavior"
                        :prop="behavior" 
                        :label="behavior" 
                        width="120"
                      />
                    </el-table>
                  </el-tab-pane>

                  <!-- 共享关系分析 -->
                  <el-tab-pane v-if="analysisResult.principal_neuron_analysis?.shared_analysis" label="共享关系" name="shared">
                    <div v-for="(data, key) in analysisResult.principal_neuron_analysis.shared_analysis" :key="key">
                      <h4>{{ key }}</h4>
                      <p>共享神经元数量: {{ data.shared_count || 0 }}</p>
                      <el-tag 
                        v-for="neuron in (data.shared || [])" 
                        :key="neuron" 
                        class="neuron-tag"
                        type="warning"
                      >
                        {{ neuron }}
                      </el-tag>
                    </div>
                  </el-tab-pane>
                </el-tabs>
              </div>
            </el-card>
          </div>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import { ref, reactive, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { 
  DataAnalysis, 
  Setting, 
  Upload, 
  UploadFilled, 
  Refresh, 
  Document, 
  Picture, 
  Grid,
  Location
} from '@element-plus/icons-vue'
import { neuronAPI } from '@/api'
import PositionMarker from '@/components/PositionMarker.vue'

export default {
  name: 'NeuronAnalysis',
  components: {
    DataAnalysis,
    Setting,
    Upload,
    UploadFilled,
    Refresh,
    Document,
    Picture,
    Grid,
    Location,
    PositionMarker
  },
  setup() {
    const loading = ref(false)
    const analysisResult = ref(null)
    const activeTab = ref('effect-size')
    const activeDataTab = ref('key-neurons')
    const uploadRef = ref(null)
    const positionUploadRef = ref(null)

    const form = reactive({
      file: null,
      positionFile: null,
      analysisType: 'comprehensive',
      behaviorColumn: '',
      threshold: 0.5,
      hasPositionData: false
    })

    const sidebarWidth = computed(() => {
      return '400px'
    })

    const handleFileChange = (file) => {
      form.file = file.raw
    }

    const handleFileRemove = () => {
      form.file = null
    }

    const handlePositionFileChange = (file) => {
      form.positionFile = file.raw
    }

    const handlePositionFileRemove = () => {
      form.positionFile = null
    }

    const handleAnalysisTypeChange = (value) => {
      if (value === 'effect-size') {
        form.hasPositionData = false
      }
    }

    const formatThresholdTooltip = (value) => {
      if (value < 0.2) return '小效应'
      if (value < 0.5) return '中等效应'
      if (value < 0.8) return '大效应'
      return '极大效应'
    }

    const getAnalysisButtonText = () => {
      if (loading.value) return '分析中...'
      switch (form.analysisType) {
        case 'effect-size': return '开始效应量分析'
        case 'principal': return '开始主神经元分析'
        case 'comprehensive': return '开始综合分析'
        default: return '开始分析'
      }
    }

    const getSummaryLabel = (key) => {
      const labels = {
        total_neurons: '总神经元数',
        total_behaviors: '总行为数',
        key_neurons_found: '关键神经元数',
        analysis_timestamp: '分析时间'
      }
      return labels[key] || key
    }

    const formatEffectSizesData = () => {
      if (!analysisResult.value?.effect_size_analysis?.effect_sizes) return []
      
      const effectSizes = analysisResult.value.effect_size_analysis.effect_sizes
      return Object.keys(effectSizes).map(neuron => {
        const row = { neuron }
        Object.keys(effectSizes[neuron]).forEach(behavior => {
          row[behavior] = effectSizes[neuron][behavior].toFixed(3)
        })
        return row
      })
    }

    const getBehaviors = () => {
      if (!analysisResult.value?.effect_size_analysis?.effect_sizes) return []
      const effectSizes = analysisResult.value.effect_size_analysis.effect_sizes
      const firstNeuron = Object.keys(effectSizes)[0]
      return firstNeuron ? Object.keys(effectSizes[firstNeuron]) : []
    }

    const getSummaryData = () => {
      if (!analysisResult.value) {
        console.log('没有分析结果数据')
        return {}
      }
      
      console.log('当前分析结果:', analysisResult.value)
      
      // 根据分析类型获取摘要数据
      if (analysisResult.value.effect_size_analysis?.statistics) {
        console.log('获取效应量分析统计:', analysisResult.value.effect_size_analysis.statistics)
        return analysisResult.value.effect_size_analysis.statistics
      } else if (analysisResult.value.principal_neuron_analysis?.statistics) {
        console.log('获取主神经元分析统计:', analysisResult.value.principal_neuron_analysis.statistics)
        return analysisResult.value.principal_neuron_analysis.statistics
      } else if (analysisResult.value.statistics) {
        console.log('获取综合分析统计:', analysisResult.value.statistics)
        return analysisResult.value.statistics
      }
      
      console.log('没有找到统计数据')
      return {}
    }

    const startAnalysis = async () => {
      if (!form.file) {
        ElMessage.warning('请先上传数据文件')
        return
      }

      loading.value = true
      let progressInterval = null

      try {
        // 创建FormData
        const formData = new FormData()
        formData.append('file', form.file)
        
        if (form.behaviorColumn) {
          formData.append('behavior_column', form.behaviorColumn)
        }
        
        formData.append('threshold', form.threshold.toString())

        // 如果有位置数据，添加到FormData
        if (form.hasPositionData && positionData.value) {
          formData.append('positions_data', JSON.stringify(positionData.value))
        }

        // 显示进度
        progressInterval = setInterval(() => {
          ElMessage.info('分析进行中，请稍候...')
        }, 5000)

        // 调用相应的API
        console.log('开始API调用，分析类型:', form.analysisType)
        console.log('FormData内容:', formData)
        console.log('文件信息:', {
          name: form.file?.name,
          size: form.file?.size,
          type: form.file?.type
        })
        
        let response
        switch (form.analysisType) {
          case 'effect-size':
            console.log('调用效应量分析API')
            response = await neuronAPI.effectSize(formData)
            break
          case 'principal':
            console.log('调用主神经元分析API')
            response = await neuronAPI.principalAnalysis(formData)
            break
          case 'comprehensive':
            console.log('调用综合分析API')
            response = await neuronAPI.comprehensiveAnalysis(formData)
            break
          default:
            throw new Error('未知的分析类型')
        }
        
        console.log('API响应:', response)
        console.log('响应类型:', typeof response)
        console.log('响应数据结构:', {
          hasSuccess: !!response.success,
          hasResult: !!response.result,
          responseKeys: Object.keys(response || {}),
          successValue: response.success
        })

        if (progressInterval) {
          clearInterval(progressInterval)
        }

        // 检查响应是否有效
        if (!response) {
          console.error('响应无效详情:', {
            response: response,
            responseType: typeof response
          })
          throw new Error('服务器响应无效')
        }

        // 检查响应格式 - 数据直接在response中，不在response.data中
        if (response.success) {
          console.log('响应结果数据:', response.result)
          console.log('分析类型:', form.analysisType)
          
          // 根据分析类型包装结果数据
          if (form.analysisType === 'effect-size') {
            analysisResult.value = {
              effect_size_analysis: response.result
            }
            console.log('包装后的效应量分析结果:', analysisResult.value)
          } else if (form.analysisType === 'principal') {
            analysisResult.value = {
              principal_neuron_analysis: response.result
            }
          } else if (form.analysisType === 'comprehensive') {
            analysisResult.value = response.result
          }
          ElMessage.success('分析完成！')
        } else {
          throw new Error(response.message || '分析失败')
        }

      } catch (error) {
        if (progressInterval) {
          clearInterval(progressInterval)
        }
        console.error('分析错误:', error)
        console.error('错误详情:', {
          message: error.message,
          response: error.response,
          status: error.response?.status,
          data: error.response?.data
        })
        
        // 根据错误类型显示不同的错误信息
        let errorMessage = '分析失败'
        if (error.response) {
          if (error.response.status === 500) {
            errorMessage = '服务器内部错误，请检查后端日志'
          } else if (error.response.status === 404) {
            errorMessage = 'API端点不存在'
          } else if (error.response.data && error.response.data.detail) {
            errorMessage = error.response.data.detail
          }
        } else if (error.message) {
          errorMessage = error.message
        }
        
        ElMessage.error(`分析失败: ${errorMessage}`)
      } finally {
        loading.value = false
      }
    }

    // 位置标记相关
    const positionMarkerRef = ref(null)
    const positionData = ref(null)
    
    const handlePositionsUpdated = (data) => {
      positionData.value = data
      console.log('位置数据已更新:', data)
    }
    
    const resetForm = () => {
      form.file = null
      form.positionFile = null
      form.analysisType = 'comprehensive'
      form.behaviorColumn = ''
      form.threshold = 0.5
      form.hasPositionData = false
      analysisResult.value = null
      positionData.value = null
      activeTab.value = 'effect-size'
      activeDataTab.value = 'key-neurons'
      
      if (uploadRef.value) {
        uploadRef.value.clearFiles()
      }
      if (positionUploadRef.value) {
        positionUploadRef.value.clearFiles()
      }
    }

    return {
      loading,
      analysisResult,
      activeTab,
      activeDataTab,
      uploadRef,
      positionUploadRef,
      form,
      sidebarWidth,
      handleFileChange,
      handleFileRemove,
      handlePositionFileChange,
      handlePositionFileRemove,
      handleAnalysisTypeChange,
      formatThresholdTooltip,
      getAnalysisButtonText,
      getSummaryLabel,
      getSummaryData,
      formatEffectSizesData,
      getBehaviors,
      startAnalysis,
      resetForm,
      positionMarkerRef,
      positionData,
      handlePositionsUpdated
    }
  }
}
</script>

<style scoped>
.neuron-analysis {
  min-height: 100vh;
  background-color: #f5f7fa;
}

.page-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
  padding: 20px;
}

.page-title {
  margin: 0;
  font-size: 28px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.page-description {
  margin: 10px 0 0 0;
  font-size: 16px;
  opacity: 0.9;
}

.parameter-panel {
  background-color: white;
  border-right: 1px solid #e4e7ed;
}

.parameter-card {
  height: 100%;
  border: none;
  box-shadow: none;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
  color: #303133;
}

.analysis-form {
  padding: 0 10px;
}

.analysis-button {
  width: 100%;
  height: 45px;
  font-size: 16px;
  font-weight: bold;
}

.results-panel {
  padding: 20px;
  background-color: #f5f7fa;
}

.empty-result-card {
  height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.analysis-results {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.summary-card,
.visualization-card,
.data-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.summary-content {
  padding: 10px 0;
}

.summary-item {
  text-align: center;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.summary-label {
  font-size: 14px;
  color: #6c757d;
  margin-bottom: 8px;
}

.summary-value {
  font-size: 24px;
  font-weight: bold;
  color: #495057;
}

.visualization-content,
.data-content {
  padding: 10px 0;
}

.image-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: center;
}

.result-image {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.neuron-tag {
  margin: 2px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .parameter-panel {
    width: 100% !important;
  }
  
  .page-title {
    font-size: 24px;
  }
  
  .summary-item {
    margin-bottom: 10px;
  }
}
</style>
