package com.example.vramcalculator

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.example.vramcalculator.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var calculator: VRAMCalculator
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        calculator = VRAMCalculator()
        setupUI()
    }
    
    private fun setupUI() {
        // Set up spinners
        val precisionOptions = arrayOf("FP16", "FP32", "BF16", "INT8", "INT4")
        val operationOptions = arrayOf("Inference", "Training", "Fine-tuning")
        
        binding.precisionSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, precisionOptions)
        binding.operationSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, operationOptions)
        
        // Set defaults
        binding.modelIdEditText.setText("mistralai/Mistral-7B-v0.1")
        binding.batchSizeEditText.setText("1")
        binding.sequenceLengthEditText.setText("2048")
        
        binding.calculateButton.setOnClickListener {
            calculateVRAM()
        }
    }
    
    private fun calculateVRAM() {
        val modelId = binding.modelIdEditText.text.toString().trim()
        if (modelId.isEmpty()) {
            Toast.makeText(this, "Please enter a model ID", Toast.LENGTH_SHORT).show()
            return
        }
        
        binding.progressBar.isVisible = true
        binding.calculateButton.isEnabled = false
        binding.resultsTextView.text = "🔍 Fetching model configuration..."
        
        calculator.getModelConfig(modelId) { result ->
            runOnUiThread {
                binding.progressBar.isVisible = false
                binding.calculateButton.isEnabled = true
                
                when (result) {
                    is Result.Success -> {
                        val modelConfig = result.data
                        performCalculation(modelConfig)
                    }
                    is Result.Failure -> {
                        binding.resultsTextView.text = "❌ Error: ${result.exception.message}\n\nUsing estimated configuration."
                        val estimatedConfig = calculator.estimateConfigFromName(modelId)
                        performCalculation(estimatedConfig)
                    }
                }
            }
        }
    }
    
    private fun performCalculation(modelConfig: ModelConfig) {
        val precision = when (binding.precisionSpinner.selectedItem as String) {
            "FP32" -> PrecisionMode.FP32
            "BF16" -> PrecisionMode.BF16
            "INT8" -> PrecisionMode.INT8
            "INT4" -> PrecisionMode.INT4
            else -> PrecisionMode.FP16
        }
        
        val operation = when (binding.operationSpinner.selectedItem as String) {
            "Training" -> OperationMode.TRAINING
            "Fine-tuning" -> OperationMode.FINE_TUNING
            else -> OperationMode.INFERENCE
        }
        
        val batchSize = binding.batchSizeEditText.text.toString().toIntOrNull() ?: 1
        val sequenceLength = binding.sequenceLengthEditText.text.toString().toIntOrNull() ?: 2048
        
        binding.resultsTextView.text = "🧮 Calculating VRAM requirements..."
        
        Thread {
            val breakdown = calculator.calculateVRAMRequirements(
                modelConfig = modelConfig,
                precision = precision,
                operation = operation,
                batchSize = batchSize,
                sequenceLength = sequenceLength
            )
            
            runOnUiThread {
                displayResults(breakdown, modelConfig)
            }
        }.start()
    }
    
    private fun displayResults(breakdown: VRAMBreakdown, modelConfig: ModelConfig) {
        val sb = StringBuilder()
        sb.append("✅ CALCULATION COMPLETE\n\n")
        sb.append("📊 MODEL INFO\n")
        sb.append("${"=".repeat(40)}\n")
        sb.append("• Model: ${modelConfig.modelId}\n")
        sb.append("• Parameters: ${modelConfig.totalParams / 1_000_000_000}B\n\n")
        
        sb.append("💾 MEMORY BREAKDOWN\n")
        sb.append("${"=".repeat(40)}\n")
        sb.append("• Model Parameters: ${"%.1f".format(breakdown.parameters.typicalGB)} GB\n")
        if (breakdown.optimizer.typicalGB > 0) sb.append("• Optimizer States: ${"%.1f".format(breakdown.optimizer.typicalGB)} GB\n")
        if (breakdown.gradients.typicalGB > 0) sb.append("• Gradients: ${"%.1f".format(breakdown.gradients.typicalGB)} GB\n")
        sb.append("• Activations: ${"%.1f".format(breakdown.activations.typicalGB)} GB\n")
        if (breakdown.kvCache.typicalGB > 0) sb.append("• KV Cache: ${"%.1f".format(breakdown.kvCache.typicalGB)} GB\n")
        sb.append("• Framework Overhead: ${"%.1f".format(breakdown.frameworkOverhead.typicalGB)} GB\n\n")
        
        sb.append("🎯 TOTAL REQUIREMENTS\n")
        sb.append("${"=".repeat(40)}\n")
        sb.append("• Typical Usage: ${"%.1f".format(breakdown.total.typicalGB)} GB\n")
        sb.append("• Realistic Range: ${"%.1f".format(breakdown.total.minimumGB)} - ${"%.1f".format(breakdown.total.maximumGB)} GB\n")
        sb.append("• Safety Buffer: +${"%.1f".format(breakdown.total.safetyBufferGB)} GB\n")
        sb.append("• Safe Target: ${"%.1f".format(breakdown.total.maximumGB + breakdown.total.safetyBufferGB)} GB\n\n")
        
        val recommendedGPU = calculator.getRecommendedGPU(breakdown.total)
        sb.append("💡 RECOMMENDED GPU\n")
        sb.append("${"=".repeat(40)}\n")
        sb.append("$recommendedGPU\n\n")
        
        sb.append("⚠️  IMPORTANT NOTES\n")
        sb.append("${"=".repeat(40)}\n")
        sb.append("• These are estimates with significant uncertainty\n")
        sb.append("• Real-world usage can vary by 2x or more\n")
        sb.append("• Always test with actual workloads\n")
        
        binding.resultsTextView.text = sb.toString()
    }
}
