#!/usr/bin/env node

// Validation script for CUDA-WASM vs WebGPU benchmarks
import puppeteer from 'puppeteer';
import fs from 'fs';

const TEST_URL = 'http://localhost:8001/test-suite.html';
const RESULTS_FILE = 'benchmark-validation-results.json';

async function runValidation() {
    console.log('🧪 Iniciando validación de benchmarks...');
    
    const browser = await puppeteer.launch({
        headless: false, // Para ver lo que pasa
        args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU']
    });
    
    try {
        const page = await browser.newPage();
        
        // Escuchar logs de consola
        page.on('console', msg => {
            console.log(`[BROWSER] ${msg.text()}`);
        });
        
        // Escuchar errores
        page.on('pageerror', error => {
            console.error(`[ERROR] ${error.message}`);
        });
        
        await page.goto(TEST_URL);
        
        // Esperar a que la página cargue
        await page.waitForSelector('#systemStatus', { timeout: 10000 });
        
        console.log('✅ Página cargada correctamente');
        
        // Ejecutar suite completa
        console.log('🚀 Ejecutando suite completa...');
        await page.click('button[onclick="runFullSuite()"]');
        
        // Esperar a que terminen todos los tests (30 segundos debería ser suficiente)
        await page.waitForTimeout(30000);
        
        // Extraer resultados
        const results = await page.evaluate(() => {
            return {
                systemStatus: document.getElementById('systemStatus').innerText,
                basicResults: document.getElementById('basicResults').innerText,
                advancedResults: document.getElementById('advancedResults').innerText,
                performanceResults: document.getElementById('performanceResults').innerText
            };
        });
        
        console.log('📊 Resultados obtenidos:');
        console.log(JSON.stringify(results, null, 2));
        
        // Guardar resultados
        fs.writeFileSync(RESULTS_FILE, JSON.stringify(results, null, 2));
        console.log(`💾 Resultados guardados en ${RESULTS_FILE}`);
        
        // Análisis simple de resultados
        analyzeResults(results);
        
    } catch (error) {
        console.error('❌ Error durante la validación:', error);
    } finally {
        await browser.close();
    }
}

function analyzeResults(results) {
    console.log('\n🔍 Análisis de resultados:');
    
    const hasWebGPU = results.systemStatus.includes('WebGPU: ✅');
    const hasWebAssembly = results.systemStatus.includes('WebAssembly: ✅');
    
    console.log(`WebGPU disponible: ${hasWebGPU ? '✅' : '❌'}`);
    console.log(`WebAssembly disponible: ${hasWebAssembly ? '✅' : '❌'}`);
    
    // Buscar errores en los resultados
    const hasErrors = results.basicResults.includes('Error') || 
                     results.advancedResults.includes('Error');
    
    console.log(`Errores detectados: ${hasErrors ? '❌' : '✅'}`);
    
    // Buscar tiempos de ejecución
    const timeMatches = (results.basicResults + results.advancedResults).match(/(\d+\.?\d*)\s*ms/g);
    if (timeMatches) {
        console.log('⏱️ Tiempos de ejecución encontrados:');
        timeMatches.forEach(time => console.log(`  - ${time}`));
    }
    
    console.log('\n📋 Resumen:');
    console.log(`- Tests ejecutados: ${hasWebGPU && hasWebAssembly ? 'Completos' : 'Parciales'}`);
    console.log(`- Estado general: ${!hasErrors ? 'Exitoso' : 'Con errores'}`);
}

// Ejecutar si es llamado directamente
if (import.meta.url === `file://${process.argv[1]}`) {
    runValidation().catch(console.error);
}