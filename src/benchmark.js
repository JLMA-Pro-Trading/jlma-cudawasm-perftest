import { WebGPUComputer } from './webgpu.js';
import { CudaWasmComputer } from './cuda-wasm.js';

class BenchmarkManager {
  constructor() {
    this.webgpu = new WebGPUComputer();
    this.cudaWasm = new CudaWasmComputer();
    this.isRunning = false;
  }

  // Generar datos de prueba
  generateTestData(size) {
    const a = new Float32Array(size);
    const b = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
      a[i] = Math.random() * 100;
      b[i] = Math.random() * 100;
    }
    
    return { a, b };
  }

  generateMatrixData(size) {
    const matrixA = new Float32Array(size * size);
    const matrixB = new Float32Array(size * size);
    
    for (let i = 0; i < size * size; i++) {
      matrixA[i] = Math.random() * 10;
      matrixB[i] = Math.random() * 10;
    }
    
    return { matrixA, matrixB };
  }

  // Actualizar interfaz
  updateProgress(percent) {
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = `${percent}%`;
  }

  updateStatus(message) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
  }

  updateResult(type, result) {
    const timeEl = document.getElementById(`${type}Time`);
    const detailsEl = document.getElementById(`${type}Details`);
    const cardEl = document.getElementById(`${type}Result`);

    if (result.error) {
      timeEl.textContent = 'Error';
      detailsEl.innerHTML = `
        <div style="color: #ff6b6b;">❌ ${result.error}</div>
      `;
      cardEl.style.borderLeftColor = '#ff6b6b';
      return;
    }

    timeEl.textContent = `${result.averageTime.toFixed(2)} ms`;
    
    detailsEl.innerHTML = `
      <div><strong>Promedio:</strong> ${result.averageTime.toFixed(2)} ms</div>
      <div><strong>Mínimo:</strong> ${result.minTime.toFixed(2)} ms</div>
      <div><strong>Máximo:</strong> ${result.maxTime.toFixed(2)} ms</div>
      <div><strong>Iteraciones:</strong> ${result.iterations}</div>
      ${result.deviceInfo ? `<div><strong>Dispositivo:</strong> ${result.deviceInfo.name}</div>` : ''}
    `;

    // Resetear estilo
    cardEl.classList.remove('winner');
    cardEl.style.borderLeftColor = '#ffd700';
  }

  showComparison(webgpuResult, cudaResult) {
    const comparisonEl = document.getElementById('comparison');
    const winnerTextEl = document.getElementById('winnerText');
    const speedupTextEl = document.getElementById('speedupText');

    if (webgpuResult.error || cudaResult.error) {
      comparisonEl.style.display = 'none';
      return;
    }

    const webgpuTime = webgpuResult.averageTime;
    const cudaTime = cudaResult.averageTime;
    
    let winner, speedup;
    
    if (webgpuTime < cudaTime) {
      winner = 'WebGPU';
      speedup = (cudaTime / webgpuTime).toFixed(2);
      document.getElementById('webgpuResult').classList.add('winner');
    } else {
      winner = 'CUDA-WASM';
      speedup = (webgpuTime / cudaTime).toFixed(2);
      document.getElementById('cudawasmResult').classList.add('winner');
    }

    winnerTextEl.innerHTML = `🏆 <strong>${winner}</strong> es el ganador!`;
    speedupTextEl.innerHTML = `⚡ Es <strong>${speedup}x</strong> más rápido que la alternativa`;
    
    comparisonEl.style.display = 'block';
  }

  // Benchmarks principales
  async runVectorAddBenchmark(size, iterations) {
    const { a, b } = this.generateTestData(size);
    
    this.updateStatus('Ejecutando WebGPU...');
    this.updateProgress(25);
    
    let webgpuResult;
    try {
      const result = await this.webgpu.vectorAdd(a, b, iterations);
      const deviceInfo = await this.getWebGPUInfo();
      webgpuResult = { ...result, deviceInfo };
    } catch (error) {
      webgpuResult = { error: error.message };
    }
    
    this.updateResult('webgpu', webgpuResult);
    this.updateProgress(50);

    this.updateStatus('Ejecutando CUDA-WASM...');
    this.updateProgress(75);
    
    let cudaResult;
    try {
      const result = await this.cudaWasm.vectorAdd(a, b, iterations);
      const deviceInfo = await this.cudaWasm.getDeviceInfo();
      cudaResult = { ...result, deviceInfo };
    } catch (error) {
      cudaResult = { error: error.message };
    }
    
    this.updateResult('cudawasm', cudaResult);
    this.updateProgress(100);

    this.showComparison(webgpuResult, cudaResult);
    this.updateStatus('✅ Benchmark completado');

    return { webgpu: webgpuResult, cudaWasm: cudaResult };
  }

  async runMatrixBenchmark(size, iterations) {
    const { matrixA, matrixB } = this.generateMatrixData(size);
    
    this.updateStatus(`Ejecutando multiplicación de matrices ${size}x${size} con WebGPU...`);
    this.updateProgress(25);
    
    let webgpuResult;
    try {
      const result = await this.webgpu.matrixMultiply(matrixA, matrixB, size, iterations);
      const deviceInfo = await this.getWebGPUInfo();
      webgpuResult = { ...result, deviceInfo };
    } catch (error) {
      webgpuResult = { error: error.message };
    }
    
    this.updateResult('webgpu', webgpuResult);
    this.updateProgress(50);

    this.updateStatus(`Ejecutando multiplicación de matrices ${size}x${size} con CUDA-WASM...`);
    this.updateProgress(75);
    
    let cudaResult;
    try {
      const result = await this.cudaWasm.matrixMultiply(matrixA, matrixB, size, iterations);
      const deviceInfo = await this.cudaWasm.getDeviceInfo();
      cudaResult = { ...result, deviceInfo };
    } catch (error) {
      cudaResult = { error: error.message };
    }
    
    this.updateResult('cudawasm', cudaResult);
    this.updateProgress(100);

    this.showComparison(webgpuResult, cudaResult);
    this.updateStatus('✅ Benchmark de matrices completado');

    return { webgpu: webgpuResult, cudaWasm: cudaResult };
  }

  async runWebGPUOnly(size, iterations) {
    const { a, b } = this.generateTestData(size);
    
    this.updateStatus('Ejecutando solo WebGPU...');
    this.updateProgress(50);
    
    try {
      const result = await this.webgpu.vectorAdd(a, b, iterations);
      const deviceInfo = await this.getWebGPUInfo();
      this.updateResult('webgpu', { ...result, deviceInfo });
      this.updateProgress(100);
      this.updateStatus('✅ WebGPU completado');
    } catch (error) {
      this.updateResult('webgpu', { error: error.message });
      this.updateStatus('❌ Error en WebGPU');
    }
  }

  async runCudaWasmOnly(size, iterations) {
    const { a, b } = this.generateTestData(size);
    
    this.updateStatus('Ejecutando solo CUDA-WASM...');
    this.updateProgress(50);
    
    try {
      const result = await this.cudaWasm.vectorAdd(a, b, iterations);
      const deviceInfo = await this.cudaWasm.getDeviceInfo();
      this.updateResult('cudawasm', { ...result, deviceInfo });
      this.updateProgress(100);
      this.updateStatus('✅ CUDA-WASM completado');
    } catch (error) {
      this.updateResult('cudawasm', { error: error.message });
      this.updateStatus('❌ Error en CUDA-WASM');
    }
  }

  async getWebGPUInfo() {
    if (!navigator.gpu) return { name: 'No disponible', type: 'N/A' };
    
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { name: 'No disponible', type: 'N/A' };
      
      const info = await adapter.requestAdapterInfo?.();
      return {
        name: info?.description || 'WebGPU',
        type: 'GPU',
        vendor: info?.vendor || 'N/A'
      };
    } catch {
      return { name: 'WebGPU', type: 'GPU' };
    }
  }

  setRunning(running) {
    this.isRunning = running;
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
      btn.disabled = running;
    });
  }
}

// Instancia global del benchmark manager
const benchmarkManager = new BenchmarkManager();

// Funciones globales para los botones
window.runBenchmark = async function() {
  if (benchmarkManager.isRunning) return;
  
  benchmarkManager.setRunning(true);
  const size = parseInt(document.getElementById('vectorSize').value);
  const iterations = parseInt(document.getElementById('iterations').value);
  
  try {
    // Limpiar resultados anteriores
    document.getElementById('comparison').style.display = 'none';
    benchmarkManager.updateProgress(0);
    
    // Decidir qué tipo de benchmark ejecutar basado en el tamaño
    if (size <= 102400) {
      // Vector addition para tamaños pequeños y medianos
      await benchmarkManager.runVectorAddBenchmark(size, iterations);
    } else {
      // Alternar entre vector add y multiplicación de matrices para tamaños grandes
      const matrixSize = Math.sqrt(size);
      if (Number.isInteger(matrixSize) && matrixSize <= 1024) {
        await benchmarkManager.runMatrixBenchmark(matrixSize, Math.max(1, Math.floor(iterations / 5)));
      } else {
        await benchmarkManager.runVectorAddBenchmark(size, iterations);
      }
    }
  } catch (error) {
    console.error('Error en benchmark:', error);
    benchmarkManager.updateStatus('❌ Error durante el benchmark');
  } finally {
    benchmarkManager.setRunning(false);
  }
};

window.runWebGPUOnly = async function() {
  if (benchmarkManager.isRunning) return;
  
  benchmarkManager.setRunning(true);
  const size = parseInt(document.getElementById('vectorSize').value);
  const iterations = parseInt(document.getElementById('iterations').value);
  
  try {
    benchmarkManager.updateProgress(0);
    await benchmarkManager.runWebGPUOnly(size, iterations);
  } catch (error) {
    console.error('Error en WebGPU:', error);
    benchmarkManager.updateStatus('❌ Error en WebGPU');
  } finally {
    benchmarkManager.setRunning(false);
  }
};

window.runCudaWasmOnly = async function() {
  if (benchmarkManager.isRunning) return;
  
  benchmarkManager.setRunning(true);
  const size = parseInt(document.getElementById('vectorSize').value);
  const iterations = parseInt(document.getElementById('iterations').value);
  
  try {
    benchmarkManager.updateProgress(0);
    await benchmarkManager.runCudaWasmOnly(size, iterations);
  } catch (error) {
    console.error('Error en CUDA-WASM:', error);
    benchmarkManager.updateStatus('❌ Error en CUDA-WASM');
  } finally {
    benchmarkManager.setRunning(false);
  }
};

// Inicializar cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 Benchmark de CUDA-WASM vs WebGPU iniciado');
  
  // Verificar soporte de WebGPU
  if (!navigator.gpu) {
    benchmarkManager.updateStatus('⚠️ WebGPU no está disponible en este navegador');
    document.getElementById('webgpuResult').querySelector('.time-display').textContent = 'No disponible';
    document.getElementById('webgpuDetails').textContent = 'WebGPU no está soportado en este navegador';
  }
});