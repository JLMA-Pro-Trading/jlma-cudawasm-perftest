// TypeScript definitions for CUDA-WASM

export interface TranspileOptions {
    output?: string;
    optimize?: boolean;
    target?: 'wasm' | 'webgpu';
    debug?: boolean;
}

export interface TranspileResult {
    success: boolean;
    inputFile: string;
    outputFile: string;
    size: number;
    optimizations: string[];
    warnings: string[];
}

export interface AnalysisResult {
    kernelName: string;
    complexity: 'low' | 'medium' | 'high';
    memoryAccess: string;
    optimization_suggestions: string[];
}

export interface BenchmarkResult {
    nativeTime: number;
    wasmTime: number;
    speedup: number;
    throughput: string;
    efficiency: string;
}

export declare function transpileCuda(inputFile: string, options?: TranspileOptions): TranspileResult;
export declare function analyzeKernel(kernelFile: string): AnalysisResult;
export declare function benchmark(kernelFile: string, options?: any): BenchmarkResult;
export declare function getVersion(): string;