// CUDA Parser - Parses CUDA code and extracts kernel information
const fs = require('fs');

class CudaParser {
    constructor() {
        this.kernels = [];
        this.globalFunctions = [];
        this.deviceFunctions = [];
    }

    parse(cudaCode) {
        // Reset state
        this.kernels = [];
        this.globalFunctions = [];
        this.deviceFunctions = [];

        // Extract kernels
        const kernelRegex = /__global__\s+(\w+\s+)?(\w+)\s*\(([^)]*)\)\s*{/g;
        let match;
        
        while ((match = kernelRegex.exec(cudaCode)) !== null) {
            const returnType = match[1] ? match[1].trim() : 'void';
            const kernelName = match[2];
            const params = this.parseParameters(match[3]);
            
            // Find the kernel body
            const startIdx = match.index + match[0].length;
            const body = this.extractBody(cudaCode, startIdx);
            
            this.kernels.push({
                name: kernelName,
                returnType,
                parameters: params,
                body,
                attributes: this.extractAttributes(cudaCode, match.index)
            });
        }

        // Extract device functions
        const deviceRegex = /__device__\s+(\w+\s+)?(\w+)\s*\(([^)]*)\)\s*{/g;
        
        while ((match = deviceRegex.exec(cudaCode)) !== null) {
            const returnType = match[1] ? match[1].trim() : 'void';
            const funcName = match[2];
            const params = this.parseParameters(match[3]);
            
            const startIdx = match.index + match[0].length;
            const body = this.extractBody(cudaCode, startIdx);
            
            this.deviceFunctions.push({
                name: funcName,
                returnType,
                parameters: params,
                body
            });
        }

        return {
            kernels: this.kernels,
            deviceFunctions: this.deviceFunctions,
            globalFunctions: this.globalFunctions
        };
    }

    parseParameters(paramStr) {
        if (!paramStr.trim()) return [];
        
        const params = [];
        const paramParts = paramStr.split(',');
        
        for (const param of paramParts) {
            const trimmed = param.trim();
            if (!trimmed) continue;
            
            // Match type and name
            const match = trimmed.match(/^(.+?)\s+(\w+)$/);
            if (match) {
                params.push({
                    type: match[1].trim(),
                    name: match[2]
                });
            }
        }
        
        return params;
    }

    extractBody(code, startIdx) {
        let braceCount = 1;
        let i = startIdx;
        let body = '';
        
        while (i < code.length && braceCount > 0) {
            const char = code[i];
            if (char === '{') braceCount++;
            else if (char === '}') braceCount--;
            
            if (braceCount > 0) {
                body += char;
            }
            i++;
        }
        
        return body.trim();
    }

    extractAttributes(code, kernelStart) {
        const attributes = [];
        
        // Look for launch bounds
        const launchBoundsRegex = /__launch_bounds__\s*\((\d+)(?:,\s*(\d+))?\)/;
        const beforeKernel = code.substring(Math.max(0, kernelStart - 200), kernelStart);
        const launchMatch = beforeKernel.match(launchBoundsRegex);
        
        if (launchMatch) {
            attributes.push({
                type: 'launch_bounds',
                maxThreads: parseInt(launchMatch[1]),
                minBlocks: launchMatch[2] ? parseInt(launchMatch[2]) : null
            });
        }
        
        return attributes;
    }

    analyzeKernel(kernel) {
        const analysis = {
            name: kernel.name,
            complexity: 'low',
            memoryPattern: 'unknown',
            threadUtilization: 0,
            sharedMemoryUsage: 0,
            registerUsage: 0,
            suggestions: []
        };

        // Analyze memory access patterns
        if (kernel.body.includes('threadIdx.x') || kernel.body.includes('blockIdx.x')) {
            analysis.memoryPattern = 'coalesced';
            analysis.threadUtilization = 80;
        }

        // Check for shared memory usage
        if (kernel.body.includes('__shared__')) {
            analysis.sharedMemoryUsage = 1024; // placeholder
            analysis.suggestions.push('Shared memory detected - ensure bank conflicts are minimized');
        }

        // Analyze complexity
        const loopCount = (kernel.body.match(/for\s*\(/g) || []).length;
        const ifCount = (kernel.body.match(/if\s*\(/g) || []).length;
        
        if (loopCount > 2 || ifCount > 3) {
            analysis.complexity = 'high';
        } else if (loopCount > 0 || ifCount > 0) {
            analysis.complexity = 'medium';
        }

        // Add suggestions based on patterns
        if (!kernel.body.includes('__syncthreads()') && kernel.body.includes('__shared__')) {
            analysis.suggestions.push('Consider adding __syncthreads() when using shared memory');
        }

        if (kernel.body.includes('malloc') || kernel.body.includes('new')) {
            analysis.suggestions.push('Dynamic memory allocation in kernels can hurt performance');
        }

        return analysis;
    }
}

module.exports = CudaParser;