// Extended benchmarks for comprehensive performance testing
class ExtendedBenchmarks {
  constructor(webgpuComputer, cudaWasmComputer) {
    this.webgpu = webgpuComputer;
    this.cudaWasm = cudaWasmComputer;
  }

  // 1. Memory Bandwidth Test
  async memoryBandwidthTest(size = 1000000) { // Reducir tamaño por defecto
    console.log('🚀 Memory Bandwidth Test');
    
    const data = new Float32Array(size).fill(1.5);
    const results = {};

    // WebGPU memory bandwidth
    try {
      const start = performance.now();
      if (this.webgpu.memoryCopy) {
        await this.webgpu.memoryCopy(data, 10);
      } else {
        // Fallback: usar vectorAdd como proxy de memory bandwidth
        const b = new Float32Array(size).fill(0);
        await this.webgpu.vectorAdd(data, b, 3);
      }
      const end = performance.now();
      const bandwidth = (size * 4 * 10) / ((end - start) / 1000) / (1024 * 1024 * 1024); // GB/s
      results.webgpu = { time: end - start, bandwidth: bandwidth.toFixed(2) };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    // CUDA-WASM memory bandwidth
    try {
      const start = performance.now();
      if (this.cudaWasm.memoryCopy) {
        await this.cudaWasm.memoryCopy(data, 10);
      } else {
        // Fallback: usar vectorAdd como proxy
        const b = new Float32Array(size).fill(0);
        await this.cudaWasm.vectorAdd(data, b, 3);
      }
      const end = performance.now();
      const bandwidth = (size * 4 * 10) / ((end - start) / 1000) / (1024 * 1024 * 1024); // GB/s
      results.cudaWasm = { time: end - start, bandwidth: bandwidth.toFixed(2) };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    return results;
  }

  // 2. Parallel Reduction Test (Sum all elements)
  async parallelReductionTest(size = 1000000) {
    console.log('🔄 Parallel Reduction Test');
    
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() * 100;
    }

    const results = {};

    // WebGPU reduction
    try {
      const start = performance.now();
      const sum = await this.webgpu.parallelSum(data);
      const end = performance.now();
      results.webgpu = { time: end - start, result: sum };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    // CUDA-WASM reduction
    try {
      const start = performance.now();
      const sum = await this.cudaWasm.parallelSum(data);
      const end = performance.now();
      results.cudaWasm = { time: end - start, result: sum };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    // JavaScript baseline
    const start = performance.now();
    const jsSum = data.reduce((a, b) => a + b, 0);
    const end = performance.now();
    results.javascript = { time: end - start, result: jsSum };

    return results;
  }

  // 3. Image Processing Test (Gaussian Blur)
  async imageProcessingTest(width = 1920, height = 1080) {
    console.log('🖼️ Image Processing Test (Gaussian Blur)');
    
    // Generate random image data (RGBA)
    const imageSize = width * height * 4;
    const imageData = new Uint8Array(imageSize);
    for (let i = 0; i < imageSize; i++) {
      imageData[i] = Math.floor(Math.random() * 256);
    }

    const results = {};

    // WebGPU image processing
    try {
      const start = performance.now();
      const processed = await this.webgpu.gaussianBlur(imageData, width, height);
      const end = performance.now();
      results.webgpu = { time: end - start, pixelsPerSecond: (width * height / ((end - start) / 1000)).toFixed(0) };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    // CUDA-WASM image processing
    try {
      const start = performance.now();
      const processed = await this.cudaWasm.gaussianBlur(imageData, width, height);
      const end = performance.now();
      results.cudaWasm = { time: end - start, pixelsPerSecond: (width * height / ((end - start) / 1000)).toFixed(0) };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    return results;
  }

  // 4. FFT (Fast Fourier Transform) Test
  async fftTest(size = 1024) {
    console.log('🌊 FFT Performance Test');
    
    // Generate complex signal data
    const realData = new Float32Array(size);
    const imagData = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
      realData[i] = Math.sin(2 * Math.PI * i / size) + 0.5 * Math.sin(4 * Math.PI * i / size);
      imagData[i] = 0;
    }

    const results = {};

    // WebGPU FFT
    try {
      const start = performance.now();
      const fftResult = await this.webgpu.fft(realData, imagData);
      const end = performance.now();
      results.webgpu = { time: end - start, samplesPerSecond: (size / ((end - start) / 1000)).toFixed(0) };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    // CUDA-WASM FFT
    try {
      const start = performance.now();
      const fftResult = await this.cudaWasm.fft(realData, imagData);
      const end = performance.now();
      results.cudaWasm = { time: end - start, samplesPerSecond: (size / ((end - start) / 1000)).toFixed(0) };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    return results;
  }

  // 5. Neural Network Inference Test
  async neuralNetworkTest() {
    console.log('🧠 Neural Network Inference Test');
    
    // Simple dense layer: input(784) -> hidden(128) -> output(10)
    const inputSize = 784; // 28x28 image
    const hiddenSize = 128;
    const outputSize = 10;
    
    const input = new Float32Array(inputSize);
    const weights1 = new Float32Array(inputSize * hiddenSize);
    const weights2 = new Float32Array(hiddenSize * outputSize);
    const bias1 = new Float32Array(hiddenSize);
    const bias2 = new Float32Array(outputSize);
    
    // Initialize with random values
    for (let i = 0; i < inputSize; i++) input[i] = Math.random();
    for (let i = 0; i < weights1.length; i++) weights1[i] = (Math.random() - 0.5) * 2;
    for (let i = 0; i < weights2.length; i++) weights2[i] = (Math.random() - 0.5) * 2;
    for (let i = 0; i < hiddenSize; i++) bias1[i] = Math.random();
    for (let i = 0; i < outputSize; i++) bias2[i] = Math.random();

    const results = {};
    const iterations = 100;

    // WebGPU neural network
    try {
      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        await this.webgpu.denseLayerInference(input, weights1, bias1, weights2, bias2);
      }
      const end = performance.now();
      const inferenceTime = (end - start) / iterations;
      results.webgpu = { 
        avgInferenceTime: inferenceTime.toFixed(2),
        inferencesPerSecond: (1000 / inferenceTime).toFixed(0)
      };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    // CUDA-WASM neural network
    try {
      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        await this.cudaWasm.denseLayerInference(input, weights1, bias1, weights2, bias2);
      }
      const end = performance.now();
      const inferenceTime = (end - start) / iterations;
      results.cudaWasm = { 
        avgInferenceTime: inferenceTime.toFixed(2),
        inferencesPerSecond: (1000 / inferenceTime).toFixed(0)
      };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    return results;
  }

  // 6. Cryptographic Operations Test
  async cryptographicTest(size = 100000) {
    console.log('🔐 Cryptographic Operations Test');
    
    const data = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(Math.random() * 0xFFFFFFFF);
    }

    const results = {};

    // SHA-256 hashing test
    try {
      const start = performance.now();
      const hash = await this.webgpu.sha256Batch(data);
      const end = performance.now();
      results.webgpu = { 
        time: end - start,
        hashesPerSecond: (size / ((end - start) / 1000)).toFixed(0)
      };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    try {
      const start = performance.now();
      const hash = await this.cudaWasm.sha256Batch(data);
      const end = performance.now();
      results.cudaWasm = { 
        time: end - start,
        hashesPerSecond: (size / ((end - start) / 1000)).toFixed(0)
      };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    return results;
  }

  // 7. Monte Carlo Simulation Test
  async monteCarloTest(samples = 10000000) {
    console.log('🎲 Monte Carlo Pi Estimation Test');
    
    const results = {};

    // WebGPU Monte Carlo
    try {
      const start = performance.now();
      const pi = await this.webgpu.monteCarloPi(samples);
      const end = performance.now();
      results.webgpu = { 
        time: end - start,
        piEstimate: pi.toFixed(6),
        samplesPerSecond: (samples / ((end - start) / 1000)).toFixed(0)
      };
    } catch (error) {
      results.webgpu = { error: error.message };
    }

    // CUDA-WASM Monte Carlo
    try {
      const start = performance.now();
      const pi = await this.cudaWasm.monteCarloPi(samples);
      const end = performance.now();
      results.cudaWasm = { 
        time: end - start,
        piEstimate: pi.toFixed(6),
        samplesPerSecond: (samples / ((end - start) / 1000)).toFixed(0)
      };
    } catch (error) {
      results.cudaWasm = { error: error.message };
    }

    // No JavaScript baseline for Monte Carlo - only WebGPU vs CUDA-WASM comparison
    return results;
  }

  // 8. Memory Access Pattern Test
  async memoryAccessPatternTest(size = 1000000) {
    console.log('📍 Memory Access Pattern Test');
    
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = i;
    }

    const results = {};

    // Test different access patterns
    const patterns = ['sequential', 'random', 'strided'];
    
    for (const pattern of patterns) {
      results[pattern] = {};
      
      // WebGPU
      try {
        const start = performance.now();
        await this.webgpu.memoryAccessPattern(data, pattern);
        const end = performance.now();
        results[pattern].webgpu = { time: end - start };
      } catch (error) {
        results[pattern].webgpu = { error: error.message };
      }

      // CUDA-WASM
      try {
        const start = performance.now();
        await this.cudaWasm.memoryAccessPattern(data, pattern);
        const end = performance.now();
        results[pattern].cudaWasm = { time: end - start };
      } catch (error) {
        results[pattern].cudaWasm = { error: error.message };
      }
    }

    return results;
  }

  // 9. Scalability Test
  async scalabilityTest() {
    console.log('📈 Scalability Test');
    
    const sizes = [1000, 10000, 100000, 1000000, 10000000];
    const results = {};

    for (const size of sizes) {
      console.log(`  Testing size: ${size}`);
      results[size] = {};
      
      const a = new Float32Array(size).fill(1);
      const b = new Float32Array(size).fill(2);

      // WebGPU
      try {
        const webgpuResult = await this.webgpu.vectorAdd(a, b, 3);
        results[size].webgpu = {
          avgTime: webgpuResult.averageTime.toFixed(2),
          throughput: (size / (webgpuResult.averageTime / 1000)).toFixed(0)
        };
      } catch (error) {
        results[size].webgpu = { error: error.message };
      }

      // CUDA-WASM
      try {
        const cudaResult = await this.cudaWasm.vectorAdd(a, b, 3);
        results[size].cudaWasm = {
          avgTime: cudaResult.averageTime.toFixed(2),
          throughput: (size / (cudaResult.averageTime / 1000)).toFixed(0)
        };
      } catch (error) {
        results[size].cudaWasm = { error: error.message };
      }
    }

    return results;
  }

  // 10. Browser Compatibility Test
  async compatibilityTest() {
    console.log('🌐 Browser Compatibility Test');
    
    const results = {
      browser: navigator.userAgent,
      webgpu: {},
      cudaWasm: {}
    };

    // Test WebGPU features
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const device = await adapter.requestDevice();
          const info = await adapter.requestAdapterInfo?.();
          
          results.webgpu = {
            supported: true,
            vendor: info?.vendor || 'Unknown',
            architecture: info?.architecture || 'Unknown',
            device: info?.device || 'Unknown',
            description: info?.description || 'Unknown'
          };
        } else {
          results.webgpu.supported = false;
          results.webgpu.reason = 'No adapter available';
        }
      } catch (error) {
        results.webgpu.supported = false;
        results.webgpu.error = error.message;
      }
    } else {
      results.webgpu.supported = false;
      results.webgpu.reason = 'WebGPU not available';
    }

    // Test WebAssembly features
    try {
      if (typeof WebAssembly !== 'undefined') {
        results.cudaWasm.supported = true;
        results.cudaWasm.features = {
          instantiate: typeof WebAssembly.instantiate !== 'undefined',
          compile: typeof WebAssembly.compile !== 'undefined',
          memory: typeof WebAssembly.Memory !== 'undefined'
        };
      } else {
        results.cudaWasm.supported = false;
        results.cudaWasm.reason = 'WebAssembly not available';
      }
    } catch (error) {
      results.cudaWasm.supported = false;
      results.cudaWasm.error = error.message;
    }

    return results;
  }

  // Convenience methods for HTML buttons - usar nombres diferentes
  async runImageProcessing() {
    return this.imageProcessingTest(1920, 1080);
  }

  async runNeuralNetwork() {
    return this.neuralNetworkTest();
  }

  async runCryptographic() {
    return this.cryptographicTest(100000);
  }

  async runMonteCarlo() {
    return this.monteCarloTest(1000000); // Reducir samples para que sea más rápido
  }
}

// Export default instance
export default ExtendedBenchmarks;
