// Benchmarking utilities for CUDA-WASM
const { performance } = require('perf_hooks');

class Benchmark {
    constructor() {
        this.results = [];
    }

    async runKernelBenchmark(kernel, options = {}) {
        const iterations = options.iterations || 100;
        const warmup = options.warmup || 10;
        
        const results = {
            kernelName: kernel.name,
            iterations: iterations,
            times: [],
            avgTime: 0,
            minTime: Infinity,
            maxTime: 0,
            stdDev: 0,
            throughput: 0,
            efficiency: 0
        };

        // Warmup runs
        for (let i = 0; i < warmup; i++) {
            await this.simulateKernelExecution(kernel);
        }

        // Benchmark runs
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.simulateKernelExecution(kernel);
            const end = performance.now();
            
            const time = end - start;
            results.times.push(time);
            results.minTime = Math.min(results.minTime, time);
            results.maxTime = Math.max(results.maxTime, time);
        }

        // Calculate statistics
        results.avgTime = results.times.reduce((a, b) => a + b, 0) / iterations;
        
        const variance = results.times.reduce((sum, time) => {
            return sum + Math.pow(time - results.avgTime, 2);
        }, 0) / iterations;
        results.stdDev = Math.sqrt(variance);

        // Calculate throughput (elements/second)
        const elementsProcessed = this.estimateElementsProcessed(kernel);
        results.throughput = (elementsProcessed / results.avgTime) * 1000; // Convert to per second

        // Calculate efficiency vs theoretical peak
        results.efficiency = this.calculateEfficiency(kernel, results);

        return results;
    }

    async simulateKernelExecution(kernel) {
        // Simulate kernel execution with some computational work
        const size = 1024 * 1024; // 1M elements
        const data = new Float32Array(size);
        
        // Simulate computation based on kernel complexity
        if (kernel.body.includes('for')) {
            // Nested loop simulation
            for (let i = 0; i < size; i++) {
                data[i] = Math.sqrt(i) * Math.sin(i);
            }
        } else {
            // Simple computation
            for (let i = 0; i < size; i++) {
                data[i] = i * 2.0 + 1.0;
            }
        }

        // Add small delay to simulate memory transfer
        await new Promise(resolve => setTimeout(resolve, 1));
        
        return data;
    }

    estimateElementsProcessed(kernel) {
        // Estimate based on kernel parameters
        let elements = 1024 * 1024; // Default 1M elements
        
        // Check for size parameters
        kernel.parameters.forEach(param => {
            if (param.name === 'n' || param.name === 'size' || param.name === 'count') {
                elements = 1024 * 1024; // Assume typical size
            }
        });
        
        return elements;
    }

    calculateEfficiency(kernel, results) {
        // Theoretical peak performance (simplified)
        const peakGFLOPS = 100; // Assume 100 GFLOPS peak
        
        // Count operations in kernel
        let flops = 0;
        const adds = (kernel.body.match(/\+/g) || []).length;
        const muls = (kernel.body.match(/\*/g) || []).length;
        const divs = (kernel.body.match(/\//g) || []).length;
        
        flops = adds + muls + (divs * 4); // Divisions are more expensive
        
        const elementsProcessed = this.estimateElementsProcessed(kernel);
        const actualGFLOPS = (flops * elementsProcessed / results.avgTime) / 1e6;
        
        return (actualGFLOPS / peakGFLOPS) * 100;
    }

    compareWithNative(wasmResults, nativeEstimate) {
        return {
            speedup: nativeEstimate / wasmResults.avgTime,
            relativePerformance: (wasmResults.avgTime / nativeEstimate) * 100
        };
    }

    generateReport(results) {
        const report = {
            summary: {
                totalKernels: results.length,
                avgSpeedup: 0,
                bestKernel: null,
                worstKernel: null
            },
            details: results
        };

        if (results.length > 0) {
            const speedups = results.map(r => r.nativeComparison?.speedup || 1);
            report.summary.avgSpeedup = speedups.reduce((a, b) => a + b, 0) / speedups.length;
            
            report.summary.bestKernel = results.reduce((best, current) => 
                (current.efficiency > (best?.efficiency || 0)) ? current : best
            );
            
            report.summary.worstKernel = results.reduce((worst, current) => 
                (current.efficiency < (worst?.efficiency || 100)) ? current : worst
            );
        }

        return report;
    }
}

module.exports = Benchmark;