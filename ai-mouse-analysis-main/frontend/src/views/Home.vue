<template>
  <div class="home">
    <!-- 欢迎区域 -->
    <div class="welcome-section card">
      <div class="welcome-content">
        <div class="welcome-text">
          <h1 class="page-title">
            <el-icon><Monitor /></el-icon>
            欢迎使用钙信号分析平台
          </h1>
          <p class="welcome-description">
            这是一个为了帮助研究人员分析钙离子荧光成像数据而设计的交互式Web应用。
            通过先进的信号处理算法和机器学习技术，为您提供准确、高效的钙信号分析解决方案。
          </p>
        </div>
        <div class="welcome-image">
          <el-icon class="brain-icon"><Monitor /></el-icon>
        </div>
      </div>
    </div>

    <!-- 功能介绍 -->
    <div class="features-section">
      <h2 class="section-title">
        <el-icon><Star /></el-icon>
        主要功能
      </h2>
      
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :md="8" :lg="8">
          <div class="feature-card card">
            <div class="feature-icon extraction">
              <el-icon><DataAnalysis /></el-icon>
            </div>
            <h3>🧪 事件提取</h3>
            <ul class="feature-list">
              <li>上传一个或多个包含钙信号数据的 Excel 文件</li>
              <li>调整参数来定义和提取钙爆发事件</li>
              <li>实时预览单个神经元的分析结果</li>
              <li>批量处理文件并下载整合后的事件特征</li>
            </ul>
            <el-button type="primary" @click="$router.push('/extraction')" class="feature-button">
              开始事件提取
              <el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="8" :lg="8">
          <div class="feature-card card">
            <div class="feature-icon clustering">
              <el-icon><PieChart /></el-icon>
            </div>
            <h3>📊 聚类分析</h3>
            <ul class="feature-list">
              <li>上传由"事件提取"页面生成的特征文件</li>
              <li>对钙爆发事件进行无监督聚类</li>
              <li>支持PCA和t-SNE降维可视化</li>
              <li>通过丰富的可视化图表探索不同事件类型的特征</li>
            </ul>
            <el-button type="success" @click="$router.push('/clustering')" class="feature-button">
              开始聚类分析
              <el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="8" :lg="8">
          <div class="feature-card card">
            <div class="feature-icon heatmap">
              <el-icon><TrendCharts /></el-icon>
            </div>
            <h3>🔥 热力图分析</h3>
            <ul class="feature-list">
              <li>分析特定行为前后时间窗口的神经元活动</li>
              <li>生成标准化热力图显示钙离子浓度变化</li>
              <li>支持同一行为或不同行为的组合分析</li>
              <li>可视化神经元活动模式和时间序列特征</li>
            </ul>
            <el-button type="warning" @click="$router.push('/heatmap')" class="feature-button">
              开始热力图分析
              <el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="8" :lg="8">
          <div class="feature-card card">
            <div class="feature-icon effect-size">
              <el-icon><Star /></el-icon>
            </div>
            <h3>⭐ 效应量分析</h3>
            <ul class="feature-list">
              <li>计算神经元活动与行为之间的效应量</li>
              <li>识别对特定行为起关键作用的神经元</li>
              <li>生成效应量分布图和热力图可视化</li>
              <li>导出关键神经元和效应量统计结果</li>
            </ul>
            <el-button type="info" @click="$router.push('/effect-size')" class="feature-button">
              开始效应量分析
              <el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="8" :lg="8">
          <div class="feature-card card">
            <div class="feature-icon position-marker">
              <el-icon><Location /></el-icon>
            </div>
            <h3>📍 位置标记</h3>
            <ul class="feature-list">
              <li>在图像上交互式标记神经元位置</li>
              <li>支持拖拽调整标记点位置</li>
              <li>自动编号管理和跳过功能</li>
              <li>导出位置坐标为CSV文件</li>
            </ul>
            <el-button type="primary" @click="$router.push('/position-marker')" class="feature-button">
              开始位置标记
              <el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="8" :lg="8">
          <div class="feature-card card">
            <div class="feature-icon principal-neuron">
              <el-icon><Monitor /></el-icon>
            </div>
            <h3>🧠 主神经元分析</h3>
            <ul class="feature-list">
              <li>识别对特定行为起关键作用的神经元</li>
              <li>分析神经元间的共享和特有关系</li>
              <li>检测神经元功能社区和模块结构</li>
              <li>在空间位置上可视化关键神经元分布</li>
            </ul>
            <el-button type="primary" @click="$router.push('/principal-neuron')" class="feature-button">
              开始主神经元分析
              <el-icon class="el-icon--right"><ArrowRight /></el-icon>
            </el-button>
          </div>
        </el-col>
      </el-row>
    </div>

    <!-- 使用流程 -->
    <div class="workflow-section card">
      <h2 class="section-title">
        <el-icon><Guide /></el-icon>
        使用流程
      </h2>
      
      <el-steps :active="-1" align-center class="workflow-steps">
        <el-step title="准备数据" icon="Document">
          <template #description>
            <div class="step-description">
              确保你的 Excel 文件格式正确，包含神经元活动的时间序列数据，
              并且包含名为 'dF' 的工作表。
            </div>
          </template>
        </el-step>
        
        <el-step title="事件提取" icon="DataAnalysis">
          <template #description>
            <div class="step-description">
              进入"事件提取"页面，上传数据并调整参数，
              运行分析并下载结果文件。
            </div>
          </template>
        </el-step>
        
        <el-step title="聚类分析" icon="PieChart">
          <template #description>
            <div class="step-description">
              进入"聚类分析"页面，上传上一步生成的特征文件，
              进行聚类和可视化分析。
            </div>
          </template>
        </el-step>
        
        <el-step title="结果下载" icon="Download">
          <template #description>
            <div class="step-description">
              下载分析结果，包含聚类标签和详细的统计信息，
              用于后续的科学研究。
            </div>
          </template>
        </el-step>
      </el-steps>
    </div>

    <!-- 技术特点 -->
    <div class="tech-features-section card">
      <h2 class="section-title">
        <el-icon><Setting /></el-icon>
        技术特点
      </h2>
      
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :md="6" :lg="6">
          <div class="tech-item">
            <el-icon class="tech-icon"><Lightning /></el-icon>
            <h4>高性能算法</h4>
            <p>采用优化的信号处理算法，快速准确地检测钙瞬变事件</p>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="6" :lg="6">
          <div class="tech-item">
            <el-icon class="tech-icon"><View /></el-icon>
            <h4>实时预览</h4>
            <p>支持参数调整的实时预览，帮助用户找到最佳分析参数</p>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="6" :lg="6">
          <div class="tech-item">
            <el-icon class="tech-icon"><Files /></el-icon>
            <h4>批量处理</h4>
            <p>支持多文件批量处理，提高大规模数据分析的效率</p>
          </div>
        </el-col>
        
        <el-col :xs="24" :sm="12" :md="6" :lg="6">
          <div class="tech-item">
            <el-icon class="tech-icon"><TrendCharts /></el-icon>
            <h4>丰富可视化</h4>
            <p>提供多种可视化方式，直观展示分析结果和数据特征</p>
          </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import {
  Monitor,
  Star,
  DataAnalysis,
  PieChart,
  ArrowRight,
  Guide,
  Setting,
  Lightning,
  View,
  Files,
  TrendCharts,
  Location
} from '@element-plus/icons-vue'
</script>

<style scoped>
.home {
  max-width: 1200px;
  margin: 0 auto;
}

.welcome-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  margin-bottom: 30px;
}

.welcome-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 40px;
}

.welcome-text {
  flex: 1;
}

.welcome-text .page-title {
  color: white;
  margin-bottom: 20px;
}

.welcome-description {
  font-size: 16px;
  line-height: 1.8;
  opacity: 0.9;
}

.welcome-image {
  flex-shrink: 0;
}

.brain-icon {
  font-size: 120px;
  opacity: 0.3;
  animation: pulse 2s ease-in-out infinite alternate;
}

@keyframes pulse {
  from {
    opacity: 0.3;
    transform: scale(1);
  }
  to {
    opacity: 0.6;
    transform: scale(1.05);
  }
}

.features-section {
  margin-bottom: 30px;
}

.feature-card {
  height: 100%;
  text-align: center;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.feature-icon {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 20px;
  font-size: 36px;
  color: white;
}

.feature-icon.extraction {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.feature-icon.clustering {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.feature-icon.heatmap {
  background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
}

.feature-icon.effect-size {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
}

.feature-icon.position-marker {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.feature-icon.principal-neuron {
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
}

.feature-card h3 {
  font-size: 20px;
  margin-bottom: 15px;
  color: #2c3e50;
}

.feature-list {
  text-align: left;
  margin-bottom: 20px;
  padding-left: 0;
  list-style: none;
}

.feature-list li {
  margin-bottom: 8px;
  padding-left: 20px;
  position: relative;
  color: #606266;
  line-height: 1.6;
}

.feature-list li::before {
  content: '✓';
  position: absolute;
  left: 0;
  color: #67c23a;
  font-weight: bold;
}

.feature-button {
  width: 100%;
}

.workflow-section {
  margin-bottom: 30px;
}

.workflow-steps {
  margin-top: 30px;
}

.step-description {
  color: #606266;
  line-height: 1.6;
  margin-top: 8px;
}

.tech-features-section {
  margin-bottom: 30px;
}

.tech-item {
  text-align: center;
  padding: 20px;
}

.tech-icon {
  font-size: 48px;
  color: #409eff;
  margin-bottom: 15px;
}

.tech-item h4 {
  font-size: 16px;
  margin-bottom: 10px;
  color: #2c3e50;
}

.tech-item p {
  color: #606266;
  line-height: 1.6;
  font-size: 14px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .welcome-content {
    flex-direction: column;
    text-align: center;
  }
  
  .brain-icon {
    font-size: 80px;
  }
  
  .workflow-steps {
    direction: ltr;
  }
  
  .tech-item {
    margin-bottom: 20px;
  }
}
</style>