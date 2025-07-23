export class WebGPUComputer {
  constructor() {
    this.device = null;
    this.adapter = null;
  }

  async initialize() {
    if (!navigator.gpu) {
      throw new Error('WebGPU no está soportado en este navegador');
    }

    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error('No se pudo obtener un adaptador WebGPU');
    }

    this.device = await this.adapter.requestDevice();
    return true;
  }

  async vectorAdd(a, b, iterations = 1) {
    if (!this.device) {
      await this.initialize();
    }

    const n = a.length;
    const results = [];

    for (let iter = 0; iter < iterations; iter++) {
      const startTime = performance.now();

      // Create buffers
      const bufferA = this.device.createBuffer({
        size: a.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const bufferB = this.device.createBuffer({
        size: b.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const bufferC = this.device.createBuffer({
        size: a.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Upload data
      this.device.queue.writeBuffer(bufferA, 0, a);
      this.device.queue.writeBuffer(bufferB, 0, b);

      // Create shader
      const shaderModule = this.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> input_a: array<f32>;
          @group(0) @binding(1) var<storage, read> input_b: array<f32>;
          @group(0) @binding(2) var<storage, read_write> output: array<f32>;

          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
              let index = global_id.x;
              if (index >= arrayLength(&output)) {
                  return;
              }
              
              output[index] = input_a[index] + input_b[index];
          }
        `
      });

      // Create bind group layout
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
      });

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: bufferA } },
          { binding: 1, resource: { buffer: bufferB } },
          { binding: 2, resource: { buffer: bufferC } },
        ]
      });

      // Create compute pipeline
      const computePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });

      // Execute computation
      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
      passEncoder.end();

      // Create staging buffer for reading results
      const stagingBuffer = this.device.createBuffer({
        size: a.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      commandEncoder.copyBufferToBuffer(bufferC, 0, stagingBuffer, 0, a.byteLength);
      this.device.queue.submit([commandEncoder.finish()]);

      // Read results
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(stagingBuffer.getMappedRange().slice());
      stagingBuffer.unmap();

      const endTime = performance.now();
      const executionTime = endTime - startTime;

      results.push({
        time: executionTime,
        result: iter === 0 ? result : null // Solo guardamos el resultado de la primera iteración
      });

      // Clean up buffers for this iteration
      bufferA.destroy();
      bufferB.destroy();
      bufferC.destroy();
      stagingBuffer.destroy();
    }

    return {
      times: results.map(r => r.time),
      averageTime: results.reduce((sum, r) => sum + r.time, 0) / results.length,
      minTime: Math.min(...results.map(r => r.time)),
      maxTime: Math.max(...results.map(r => r.time)),
      result: results[0].result,
      iterations: iterations
    };
  }

  async matrixMultiply(matrixA, matrixB, size, iterations = 1) {
    if (!this.device) {
      await this.initialize();
    }

    // SAFETY CHECK: Prevent memory exhaustion crashes
    const matrixSizeBytes = size * size * 4; // 4 bytes per float
    const totalMemoryNeeded = matrixSizeBytes * 4; // A, B, C, staging buffers
    const maxAllowedMemory = 512 * 1024 * 1024; // 512MB limit
    
    if (totalMemoryNeeded > maxAllowedMemory) {
      console.warn(`⚠️ Matrix too large for WebGPU: ${size}x${size} (${(totalMemoryNeeded/1024/1024).toFixed(1)}MB)`);
      throw new Error(`Matrix size ${size}x${size} exceeds memory limit (${(maxAllowedMemory/1024/1024)}MB). Use smaller matrices.`);
    }
    
    // Additional safety for very large matrices
    if (size > 2048) {
      console.warn(`⚠️ Large matrix detected: ${size}x${size}, may cause performance issues`);
    }

    const results = [];

    for (let iter = 0; iter < iterations; iter++) {
      const startTime = performance.now();

      // Create buffers
      const bufferA = this.device.createBuffer({
        size: matrixA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const bufferB = this.device.createBuffer({
        size: matrixB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const bufferC = this.device.createBuffer({
        size: matrixA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Upload data
      this.device.queue.writeBuffer(bufferA, 0, matrixA);
      this.device.queue.writeBuffer(bufferB, 0, matrixB);

      // Create shader for matrix multiplication
      const shaderModule = this.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
          @group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
          @group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;

          @compute @workgroup_size(16, 16)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
              let row = global_id.x;
              let col = global_id.y;
              let size = ${size}u;
              
              if (row >= size || col >= size) {
                  return;
              }
              
              var sum: f32 = 0.0;
              for (var k = 0u; k < size; k = k + 1u) {
                  sum = sum + matrix_a[row * size + k] * matrix_b[k * size + col];
              }
              
              matrix_c[row * size + col] = sum;
          }
        `
      });

      // Create bind group layout
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
      });

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: bufferA } },
          { binding: 1, resource: { buffer: bufferB } },
          { binding: 2, resource: { buffer: bufferC } },
        ]
      });

      // Create compute pipeline
      const computePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });

      // Execute computation
      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(size / 16), Math.ceil(size / 16));
      passEncoder.end();

      // Create staging buffer for reading results
      const stagingBuffer = this.device.createBuffer({
        size: matrixA.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      commandEncoder.copyBufferToBuffer(bufferC, 0, stagingBuffer, 0, matrixA.byteLength);
      this.device.queue.submit([commandEncoder.finish()]);

      // Read results
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(stagingBuffer.getMappedRange().slice());
      stagingBuffer.unmap();

      const endTime = performance.now();
      const executionTime = endTime - startTime;

      results.push({
        time: executionTime,
        result: iter === 0 ? result : null
      });

      // Clean up buffers for this iteration
      bufferA.destroy();
      bufferB.destroy();
      bufferC.destroy();
      stagingBuffer.destroy();
    }

    return {
      times: results.map(r => r.time),
      averageTime: results.reduce((sum, r) => sum + r.time, 0) / results.length,
      minTime: Math.min(...results.map(r => r.time)),
      maxTime: Math.max(...results.map(r => r.time)),
      result: results[0].result,
      iterations: iterations
    };
  }

  // Extended benchmark methods
  async memoryCopy(data, iterations = 1) {
    if (!this.device) await this.initialize();
    
    const results = [];
    for (let i = 0; i < iterations; i++) {
      const buffer = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buffer, 0, data);
      await this.device.queue.onSubmittedWorkDone();
      buffer.destroy();
    }
  }

  async parallelSum(data) {
    if (!this.device) await this.initialize();
    
    // Real parallel reduction using WebGPU compute shader
    const dataSize = data.length;
    const bufferSize = dataSize * 4; // 4 bytes per float
    
    // Create input buffer
    const inputBuffer = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    this.device.queue.writeBuffer(inputBuffer, 0, data);
    
    // Parallel reduction using workgroup shared memory
    const shaderModule = this.device.createShaderModule({
      code: `
        var<workgroup> shared_data: array<f32, 256>;
        
        @group(0) @binding(0) var<storage, read> input_data: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

        @compute @workgroup_size(256)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_index) local_index: u32,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let data_size = ${dataSize}u;
            let tid = local_index;
            let i = workgroup_id.x * 512u + tid;
            
            // Load data into shared memory with coalesced access
            var sum = 0.0;
            if (i < data_size) {
                sum += input_data[i];
            }
            if (i + 256u < data_size) {
                sum += input_data[i + 256u];
            }
            shared_data[tid] = sum;
            
            workgroupBarrier();
            
            // Perform reduction in shared memory
            if (tid < 128u) { shared_data[tid] += shared_data[tid + 128u]; }
            workgroupBarrier();
            if (tid < 64u) { shared_data[tid] += shared_data[tid + 64u]; }
            workgroupBarrier();
            if (tid < 32u) { shared_data[tid] += shared_data[tid + 32u]; }
            workgroupBarrier();
            if (tid < 16u) { shared_data[tid] += shared_data[tid + 16u]; }
            workgroupBarrier();
            if (tid < 8u) { shared_data[tid] += shared_data[tid + 8u]; }
            workgroupBarrier();
            if (tid < 4u) { shared_data[tid] += shared_data[tid + 4u]; }
            workgroupBarrier();
            if (tid < 2u) { shared_data[tid] += shared_data[tid + 2u]; }
            workgroupBarrier();
            
            // Write result
            if (tid == 0u) {
                output_data[workgroup_id.x] = shared_data[0] + shared_data[1];
            }
        }
      `
    });
    
    const numWorkgroups = Math.ceil(dataSize / 512);
    const resultBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });
    
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: resultBuffer } },
      ]
    });
    
    const computePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' }
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
    
    const stagingBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();
    
    // Sum the partial results
    let totalSum = 0;
    for (let i = 0; i < results.length; i++) {
      totalSum += results[i];
    }
    
    // Cleanup
    inputBuffer.destroy();
    resultBuffer.destroy();
    stagingBuffer.destroy();
    
    return totalSum;
  }

  async gaussianBlur(imageData, width, height) {
    if (!this.device) await this.initialize();
    
    // Real Gaussian blur implementation using WebGPU compute shader
    const dataSize = width * height * 4; // RGBA
    
    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: dataSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const outputBuffer = this.device.createBuffer({
      size: dataSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Upload image data
    this.device.queue.writeBuffer(inputBuffer, 0, imageData);
    
    // Create Gaussian blur compute shader
    const shaderModule = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input_image: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output_image: array<u32>;

        // Gaussian kernel 3x3 (normalized)
        const kernel: array<f32, 9> = array<f32, 9>(
          0.0625, 0.125, 0.0625,
          0.125,  0.25,  0.125,
          0.0625, 0.125, 0.0625
        );

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = i32(global_id.x);
            let y = i32(global_id.y);
            let width = ${width};
            let height = ${height};
            
            if (x >= width || y >= height) { return; }
            
            var r: f32 = 0.0;
            var g: f32 = 0.0;
            var b: f32 = 0.0;
            var a: f32 = 0.0;
            
            // Apply 3x3 Gaussian kernel
            for (var ky = -1; ky <= 1; ky = ky + 1) {
                for (var kx = -1; kx <= 1; kx = kx + 1) {
                    let nx = clamp(x + kx, 0, width - 1);
                    let ny = clamp(y + ky, 0, height - 1);
                    let pixel_idx = ny * width + nx;
                    let pixel = input_image[pixel_idx];
                    
                    let kernel_idx = (ky + 1) * 3 + (kx + 1);
                    let weight = kernel[kernel_idx];
                    
                    r += weight * f32((pixel >> 0u) & 0xFFu);
                    g += weight * f32((pixel >> 8u) & 0xFFu);
                    b += weight * f32((pixel >> 16u) & 0xFFu);
                    a += weight * f32((pixel >> 24u) & 0xFFu);
                }
            }
            
            let output_idx = y * width + x;
            let result = (u32(a) << 24u) | (u32(b) << 16u) | (u32(g) << 8u) | u32(r);
            output_image[output_idx] = result;
        }
      `
    });
    
    // Execute blur
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });
    
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
      ]
    });
    
    const computePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' }
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    passEncoder.end();
    
    // Read results
    const stagingBuffer = this.device.createBuffer({
      size: dataSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, dataSize);
    this.device.queue.submit([commandEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint8Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();
    
    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
    
    return result;
  }

  async fft(realData, imagData) {
    if (!this.device) await this.initialize();
    
    // Real Cooley-Tukey FFT implementation using WebGPU
    const N = realData.length;
    
    // Ensure N is a power of 2
    const logN = Math.log2(N);
    if (logN !== Math.floor(logN)) {
      throw new Error('FFT size must be a power of 2');
    }
    
    // Create buffers for complex data (interleaved real/imaginary)
    const complexSize = N * 2 * 4; // N complex numbers, 2 floats each, 4 bytes per float
    
    const inputBuffer = this.device.createBuffer({
      size: complexSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const outputBuffer = this.device.createBuffer({
      size: complexSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Interleave real and imaginary data
    const complexData = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) {
      complexData[i * 2] = realData[i];
      complexData[i * 2 + 1] = imagData[i];
    }
    
    this.device.queue.writeBuffer(inputBuffer, 0, complexData);
    
    // Perform FFT stages
    let currentInput = inputBuffer;
    let currentOutput = outputBuffer;
    
    for (let stage = 1; stage <= logN; stage++) {
      const stageSize = 1 << stage; // 2^stage
      const halfStageSize = stageSize >> 1;
      
      const shaderModule = this.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> input_data: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
              let N = ${N}u;
              let stage_size = ${stageSize}u;
              let half_stage_size = ${halfStageSize}u;
              let idx = global_id.x;
              
              if (idx >= N / 2u) { return; }
              
              let block = idx / half_stage_size;
              let block_idx = idx % half_stage_size;
              
              let i = block * stage_size + block_idx;
              let j = i + half_stage_size;
              
              // Twiddle factor
              let angle = -2.0 * 3.14159265359 * f32(block_idx) / f32(stage_size);
              let wr = cos(angle);
              let wi = sin(angle);
              
              // Load complex numbers
              let ar = input_data[i * 2u];
              let ai = input_data[i * 2u + 1u];
              let br = input_data[j * 2u];
              let bi = input_data[j * 2u + 1u];
              
              // Apply twiddle factor to second element
              let tr = br * wr - bi * wi;
              let ti = br * wi + bi * wr;
              
              // Butterfly operation
              output_data[i * 2u] = ar + tr;
              output_data[i * 2u + 1u] = ai + ti;
              output_data[j * 2u] = ar - tr;
              output_data[j * 2u + 1u] = ai - ti;
          }
        `
      });
      
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
      });
      
      const bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: currentInput } },
          { binding: 1, resource: { buffer: currentOutput } },
        ]
      });
      
      const computePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' }
      });
      
      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(N / 2 / 256));
      passEncoder.end();
      this.device.queue.submit([commandEncoder.finish()]);
      
      await this.device.queue.onSubmittedWorkDone();
      
      // Swap buffers for next stage
      [currentInput, currentOutput] = [currentOutput, currentInput];
    }
    
    // Read results
    const stagingBuffer = this.device.createBuffer({
      size: complexSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(currentInput, 0, stagingBuffer, 0, complexSize);
    this.device.queue.submit([commandEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();
    
    // De-interleave real and imaginary parts
    const realResult = new Float32Array(N);
    const imagResult = new Float32Array(N);
    
    for (let i = 0; i < N; i++) {
      realResult[i] = result[i * 2];
      imagResult[i] = result[i * 2 + 1];
    }
    
    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
    
    return { real: realResult, imag: imagResult };
  }

  async denseLayerInference(input, weights1, bias1, weights2, bias2) {
    if (!this.device) await this.initialize();
    
    // Real neural network inference using WebGPU compute shader
    const inputSize = input.length;
    const hiddenSize = bias1.length;
    const outputSize = bias2.length;
    
    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const weights1Buffer = this.device.createBuffer({
      size: weights1.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const bias1Buffer = this.device.createBuffer({
      size: bias1.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    const hiddenBuffer = this.device.createBuffer({
      size: hiddenSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    const outputBuffer = this.device.createBuffer({
      size: outputSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Upload data
    this.device.queue.writeBuffer(inputBuffer, 0, input);
    this.device.queue.writeBuffer(weights1Buffer, 0, weights1);
    this.device.queue.writeBuffer(bias1Buffer, 0, bias1);
    
    // Create compute shader for first layer (input -> hidden)
    const shaderModule = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input_data: array<f32>;
        @group(0) @binding(1) var<storage, read> weights: array<f32>;
        @group(0) @binding(2) var<storage, read> bias: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output_data: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= ${hiddenSize}u) { return; }
            
            var sum: f32 = bias[idx];
            for (var i = 0u; i < ${inputSize}u; i = i + 1u) {
                sum = sum + input_data[i] * weights[i * ${hiddenSize}u + idx];
            }
            
            // ReLU activation
            output_data[idx] = max(0.0, sum);
        }
      `
    });
    
    // Execute first layer
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });
    
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weights1Buffer } },
        { binding: 2, resource: { buffer: bias1Buffer } },
        { binding: 3, resource: { buffer: hiddenBuffer } },
      ]
    });
    
    const computePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' }
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(hiddenSize / 64));
    passEncoder.end();
    
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    
    // For simplicity, do second layer on CPU (hidden -> output)
    const stagingBuffer = this.device.createBuffer({
      size: hiddenSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(hiddenBuffer, 0, stagingBuffer, 0, hiddenSize * 4);
    this.device.queue.submit([copyEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const hiddenData = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();
    
    // Second layer computation on CPU
    const output = new Float32Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
      let sum = bias2[i];
      for (let j = 0; j < hiddenSize; j++) {
        sum += hiddenData[j] * weights2[j * outputSize + i];
      }
      output[i] = sum; // Linear output
    }
    
    // Cleanup
    inputBuffer.destroy();
    weights1Buffer.destroy();
    bias1Buffer.destroy();
    hiddenBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
    
    return output;
  }

  async sha256Batch(data) {
    if (!this.device) await this.initialize();
    
    // Real SHA-256 implementation using WebCrypto API (hardware accelerated)
    // This is more realistic than trying to implement SHA-256 in WebGPU
    const hashes = [];
    
    for (let i = 0; i < data.length; i += 16) { // Process in chunks
      const chunk = data.slice(i, Math.min(i + 16, data.length));
      const buffer = new ArrayBuffer(chunk.length * 4);
      const view = new DataView(buffer);
      
      for (let j = 0; j < chunk.length; j++) {
        view.setUint32(j * 4, chunk[j], false);
      }
      
      const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
      hashes.push(new Uint8Array(hashBuffer));
    }
    
    // Return concatenated hashes (for benchmarking purposes)
    const totalLength = hashes.reduce((sum, hash) => sum + hash.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    
    for (const hash of hashes) {
      result.set(hash, offset);
      offset += hash.length;
    }
    
    return result;
  }

  async monteCarloPi(samples) {
    if (!this.device) await this.initialize();
    
    // Real Monte Carlo Pi estimation using WebGPU compute shader
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(samples / workgroupSize);
    const actualSamples = numWorkgroups * workgroupSize;
    
    // Create buffer for results
    const resultBuffer = this.device.createBuffer({
      size: numWorkgroups * 4, // One uint32 per workgroup
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    // Create compute shader
    const shaderModule = this.device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> results: array<u32>;
        
        var<workgroup> shared_count: u32;
        
        // Simple LCG random number generator
        fn random(seed: u32) -> f32 {
            let a = 1664525u;
            let c = 1013904223u;
            let m = 4294967296u; // 2^32
            let x = (a * seed + c) % m;
            return f32(x) / f32(m);
        }
        
        @compute @workgroup_size(${workgroupSize})
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_index) local_index: u32,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let sample_id = global_id.x;
            
            // Initialize shared memory
            if (local_index == 0u) {
                shared_count = 0u;
            }
            workgroupBarrier();
            
            if (sample_id < ${actualSamples}u) {
                // Generate pseudo-random point
                let seed1 = sample_id * 2u + 1u;
                let seed2 = sample_id * 2u + 2u;
                
                let x = random(seed1 + workgroup_id.x);
                let y = random(seed2 + workgroup_id.x * 37u);
                
                // Check if point is inside unit circle
                if (x * x + y * y <= 1.0) {
                    atomicAdd(&shared_count, 1u);
                }
            }
            
            workgroupBarrier();
            
            // Store workgroup result
            if (local_index == 0u) {
                results[workgroup_id.x] = shared_count;
            }
        }
      `
    });
    
    // Create bind group
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });
    
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: resultBuffer } },
      ]
    });
    
    const computePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' }
    });
    
    // Execute compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
    
    // Read results
    const stagingBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const results = new Uint32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();
    
    // Sum all results
    let totalInside = 0;
    for (let i = 0; i < results.length; i++) {
      totalInside += results[i];
    }
    
    // Cleanup
    resultBuffer.destroy();
    stagingBuffer.destroy();
    
    return (totalInside / actualSamples) * 4;
  }

  async memoryAccessPattern(data, pattern) {
    if (!this.device) await this.initialize();
    
    // Memory access pattern test
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    await this.device.queue.onSubmittedWorkDone();
    buffer.destroy();
  }
}