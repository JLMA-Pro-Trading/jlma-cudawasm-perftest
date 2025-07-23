// WebAssembly Code Generator - Converts parsed CUDA to WASM
const fs = require('fs');

class WasmGenerator {
    constructor() {
        this.wasm = {
            version: 1,
            types: [],
            functions: [],
            memory: {
                initial: 256, // 256 pages = 16MB
                maximum: 1024 // 1024 pages = 64MB
            },
            exports: []
        };
    }

    generate(parsedCuda) {
        // Reset state
        this.wasm.types = [];
        this.wasm.functions = [];
        this.wasm.exports = [];

        // Convert each kernel to WASM function
        for (const kernel of parsedCuda.kernels) {
            this.convertKernel(kernel);
        }

        // Convert device functions
        for (const func of parsedCuda.deviceFunctions) {
            this.convertDeviceFunction(func);
        }

        return this.generateWAT();
    }

    convertKernel(kernel) {
        // Create function type
        const typeIdx = this.addFunctionType(kernel.parameters, kernel.returnType);
        
        // Generate function body
        const funcBody = this.generateFunctionBody(kernel);
        
        // Add function
        const funcIdx = this.wasm.functions.length;
        this.wasm.functions.push({
            name: kernel.name,
            typeIdx,
            locals: this.extractLocals(kernel.body),
            body: funcBody
        });

        // Export kernel
        this.wasm.exports.push({
            name: kernel.name,
            kind: 'func',
            index: funcIdx
        });
    }

    convertDeviceFunction(func) {
        const typeIdx = this.addFunctionType(func.parameters, func.returnType);
        const funcBody = this.generateFunctionBody(func);
        
        this.wasm.functions.push({
            name: func.name,
            typeIdx,
            locals: this.extractLocals(func.body),
            body: funcBody
        });
    }

    addFunctionType(params, returnType) {
        const paramTypes = params.map(p => this.cudaTypeToWasm(p.type));
        const resultTypes = returnType === 'void' ? [] : [this.cudaTypeToWasm(returnType)];
        
        // Check if type already exists
        for (let i = 0; i < this.wasm.types.length; i++) {
            const type = this.wasm.types[i];
            if (JSON.stringify(type.params) === JSON.stringify(paramTypes) &&
                JSON.stringify(type.results) === JSON.stringify(resultTypes)) {
                return i;
            }
        }
        
        // Add new type
        this.wasm.types.push({
            params: paramTypes,
            results: resultTypes
        });
        
        return this.wasm.types.length - 1;
    }

    cudaTypeToWasm(cudaType) {
        const typeMap = {
            'float': 'f32',
            'double': 'f64',
            'int': 'i32',
            'unsigned int': 'i32',
            'uint': 'i32',
            'long': 'i64',
            'unsigned long': 'i64',
            'char': 'i32',
            'unsigned char': 'i32',
            'short': 'i32',
            'unsigned short': 'i32',
            'bool': 'i32'
        };

        // Handle pointers
        if (cudaType.includes('*')) {
            return 'i32'; // Pointers are 32-bit indices in WASM
        }

        return typeMap[cudaType.trim()] || 'i32';
    }

    extractLocals(body) {
        const locals = [];
        
        // Simple extraction of local variables
        const varRegex = /\b(int|float|double|char|short|long|unsigned|bool)\s+(\w+)\s*[=;]/g;
        let match;
        
        while ((match = varRegex.exec(body)) !== null) {
            locals.push({
                name: match[2],
                type: this.cudaTypeToWasm(match[1])
            });
        }
        
        return locals;
    }

    generateFunctionBody(kernel) {
        const instructions = [];
        
        // Parse kernel body and generate WASM instructions
        const lines = kernel.body.split('\n');
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            
            // Handle thread index calculations
            if (trimmed.includes('threadIdx.x')) {
                instructions.push('global.get $threadIdx_x');
            }
            if (trimmed.includes('blockIdx.x')) {
                instructions.push('global.get $blockIdx_x');
            }
            if (trimmed.includes('blockDim.x')) {
                instructions.push('global.get $blockDim_x');
            }
            
            // Handle array access
            const arrayMatch = trimmed.match(/(\w+)\[([^\]]+)\]\s*=\s*([^;]+)/);
            if (arrayMatch) {
                const array = arrayMatch[1];
                const index = arrayMatch[2];
                const value = arrayMatch[3];
                
                // Generate load/store instructions
                instructions.push(`local.get $${array}`);
                instructions.push(`${this.parseExpression(index)}`);
                instructions.push('i32.const 4'); // sizeof(float)
                instructions.push('i32.mul');
                instructions.push('i32.add');
                instructions.push(`${this.parseExpression(value)}`);
                instructions.push('f32.store');
            }
        }
        
        return instructions;
    }

    parseExpression(expr) {
        // Simple expression parser
        if (expr.includes('+')) {
            const parts = expr.split('+');
            return `${this.parseExpression(parts[0].trim())} ${this.parseExpression(parts[1].trim())} f32.add`;
        }
        
        if (expr.includes('*')) {
            const parts = expr.split('*');
            return `${this.parseExpression(parts[0].trim())} ${this.parseExpression(parts[1].trim())} f32.mul`;
        }
        
        // Variable reference
        if (expr.match(/^\w+$/)) {
            return `local.get $${expr}`;
        }
        
        // Constant
        if (!isNaN(expr)) {
            return `f32.const ${expr}`;
        }
        
        return '';
    }

    generateWAT() {
        let wat = '(module\n';
        
        // Add memory
        wat += `  (memory $mem ${this.wasm.memory.initial} ${this.wasm.memory.maximum})\n`;
        wat += '  (export "memory" (memory $mem))\n\n';
        
        // Add globals for thread/block info
        wat += '  (global $threadIdx_x (mut i32) (i32.const 0))\n';
        wat += '  (global $blockIdx_x (mut i32) (i32.const 0))\n';
        wat += '  (global $blockDim_x (mut i32) (i32.const 256))\n\n';
        
        // Add types
        this.wasm.types.forEach((type, idx) => {
            wat += `  (type $t${idx} (func`;
            if (type.params.length > 0) {
                wat += ' (param';
                type.params.forEach(p => wat += ` ${p}`);
                wat += ')';
            }
            if (type.results.length > 0) {
                wat += ' (result';
                type.results.forEach(r => wat += ` ${r}`);
                wat += ')';
            }
            wat += '))\n';
        });
        wat += '\n';
        
        // Add functions
        this.wasm.functions.forEach((func, idx) => {
            wat += `  (func $${func.name} (type $t${func.typeIdx})`;
            
            // Add locals
            if (func.locals.length > 0) {
                func.locals.forEach(local => {
                    wat += `\n    (local $${local.name} ${local.type})`;
                });
            }
            
            // Add body
            wat += '\n';
            func.body.forEach(inst => {
                wat += `    ${inst}\n`;
            });
            
            wat += '  )\n\n';
        });
        
        // Add exports
        this.wasm.exports.forEach(exp => {
            wat += `  (export "${exp.name}" (func $${exp.name}))\n`;
        });
        
        wat += ')\n';
        
        return wat;
    }

    generateBinary(wat) {
        // In a real implementation, this would use wabt or similar to convert WAT to WASM
        // For now, return the WAT text
        return Buffer.from(wat);
    }
}

module.exports = WasmGenerator;