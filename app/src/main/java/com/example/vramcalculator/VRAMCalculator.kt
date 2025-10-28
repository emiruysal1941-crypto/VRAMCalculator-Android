package com.example.vramcalculator

import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

class VRAMCalculator {
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()
    
    private val knownModels = mapOf(
        "mistralai/Mistral-7B-v0.1" to ModelConfig(
            modelId = "mistralai/Mistral-7B-v0.1",
            totalParams = 7_000_000_000,
            hiddenSize = 4096,
            numHiddenLayers = 32,
            numAttentionHeads = 32,
            numKeyValueHeads = 8,
            intermediateSize = 14336,
            vocabSize = 32000,
            modelType = "mistral"
        ),
        "google/flan-t5-large" to ModelConfig(
            modelId = "google/flan-t5-large",
            totalParams = 770_000_000,
            hiddenSize = 1024,
            numHiddenLayers = 24,
            numAttentionHeads = 16,
            numKeyValueHeads = 16,
            intermediateSize = 2816,
            vocabSize = 32128,
            modelType = "t5"
        ),
        "bert-base-uncased" to ModelConfig(
            modelId = "bert-base-uncased",
            totalParams = 110_000_000,
            hiddenSize = 768,
            numHiddenLayers = 12,
            numAttentionHeads = 12,
            numKeyValueHeads = 12,
            intermediateSize = 3072,
            vocabSize = 30522,
            modelType = "bert"
        )
    )
    
    fun getModelConfig(modelId: String, callback: (Result<ModelConfig>) -> Unit) {
        knownModels[modelId]?.let {
            callback(Result.success(it))
            return
        }
        
        val url = "https://huggingface.co/$modelId/raw/main/config.json"
        val request = Request.Builder().url(url).build()
        
        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                callback(Result.success(estimateConfigFromName(modelId)))
            }
            
            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                try {
                    if (response.isSuccessful) {
                        val configData = JSONObject(response.body?.string() ?: "{}")
                        callback(Result.success(parseModelConfig(modelId, configData)))
                    } else {
                        callback(Result.success(estimateConfigFromName(modelId)))
                    }
                } catch (e: Exception) {
                    callback(Result.success(estimateConfigFromName(modelId)))
                }
            }
        })
    }
    
    private fun parseModelConfig(modelId: String, configData: JSONObject): ModelConfig {
        val totalParams = estimateParamsFromName(modelId)
        
        return ModelConfig(
            modelId = modelId,
            totalParams = totalParams,
            hiddenSize = configData.optInt("hidden_size").takeIf { it != 0 } ?: estimateHiddenSize(totalParams),
            numHiddenLayers = configData.optInt("num_hidden_layers").takeIf { it != 0 } ?: estimateNumLayers(totalParams),
            numAttentionHeads = configData.optInt("num_attention_heads").takeIf { it != 0 } ?: estimateNumHeads(totalParams),
            numKeyValueHeads = configData.optInt("num_key_value_heads").takeIf { it != 0 } ?: estimateNumKVHeads(totalParams),
            intermediateSize = configData.optInt("intermediate_size").takeIf { it != 0 } ?: estimateIntermediateSize(totalParams),
            vocabSize = configData.optInt("vocab_size").takeIf { it != 0 },
            modelType = configData.optString("model_type")
        )
    }
    
    fun estimateConfigFromName(modelId: String): ModelConfig {
        val totalParams = estimateParamsFromName(modelId)
        return ModelConfig(
            modelId = modelId,
            totalParams = totalParams,
            hiddenSize = estimateHiddenSize(totalParams),
            numHiddenLayers = estimateNumLayers(totalParams),
            numAttentionHeads = estimateNumHeads(totalParams),
            numKeyValueHeads = estimateNumKVHeads(totalParams),
            intermediateSize = estimateIntermediateSize(totalParams),
            modelType = detectModelType(modelId)
        )
    }
    
    private fun estimateParamsFromName(modelId: String): Long {
        return when {
            "70b" in modelId.lowercase() -> 70_000_000_000
            "13b" in modelId.lowercase() -> 13_000_000_000
            "7b" in modelId.lowercase() -> 7_000_000_000
            "3b" in modelId.lowercase() -> 3_000_000_000
            "1.3b" in modelId.lowercase() -> 1_300_000_000
            "large" in modelId.lowercase() -> 350_000_000
            "base" in modelId.lowercase() -> 110_000_000
            else -> 125_000_000
        }
    }
    
    private fun detectModelType(modelId: String): String {
        val lowerId = modelId.lowercase()
        return when {
            "llama" in lowerId -> "llama"
            "mistral" in lowerId -> "mistral"
            "qwen" in lowerId -> "qwen"
            "t5" in lowerId -> "t5"
            "bert" in lowerId -> "bert"
            "gpt" in lowerId -> "gpt"
            else -> "transformer"
        }
    }
    
    private fun estimateHiddenSize(totalParams: Long): Int = when {
        totalParams > 50_000_000_000 -> 8192
        totalParams > 10_000_000_000 -> 5120
        totalParams > 1_000_000_000 -> 4096
        totalParams > 500_000_000 -> 2048
        totalParams > 100_000_000 -> 1024
        else -> 768
    }
    
    private fun estimateNumLayers(totalParams: Long): Int = when {
        totalParams > 50_000_000_000 -> 80
        totalParams > 10_000_000_000 -> 48
        totalParams > 1_000_000_000 -> 32
        totalParams > 500_000_000 -> 24
        totalParams > 100_000_000 -> 12
        else -> 6
    }
    
    private fun estimateNumHeads(totalParams: Long): Int {
        val hiddenSize = estimateHiddenSize(totalParams)
        return when {
            hiddenSize % 128 == 0 -> hiddenSize / 128
            hiddenSize % 64 == 0 -> hiddenSize / 64
            else -> 12
        }
    }
    
    private fun estimateNumKVHeads(totalParams: Long): Int {
        return maxOf(1, estimateNumHeads(totalParams) / 8)
    }
    
    private fun estimateIntermediateSize(totalParams: Long): Int {
        return estimateHiddenSize(totalParams) * 4
    }
    
    fun calculateVRAMRequirements(
        modelConfig: ModelConfig,
        precision: PrecisionMode,
        operation: OperationMode,
        batchSize: Int = 1,
        sequenceLength: Int = 2048
    ): VRAMBreakdown {
        
        val paramMem = calculateParameterMemory(modelConfig, precision)
        val optimizerMem = calculateOptimizerMemory(modelConfig, precision, operation)
        val gradientMem = calculateGradientMemory(modelConfig, precision, operation)
        val activationMem = calculateActivationMemory(modelConfig, precision, operation, batchSize, sequenceLength)
        val kvCacheMem = calculateKVCacheMemory(modelConfig, precision, operation, batchSize, sequenceLength)
        val frameworkMem = calculateFrameworkOverhead(modelConfig, precision, operation)
        
        val totalMem = calculateTotalMemory(listOf(
            paramMem, optimizerMem, gradientMem, activationMem, kvCacheMem, frameworkMem
        ))
        
        return VRAMBreakdown(
            parameters = paramMem,
            optimizer = optimizerMem,
            gradients = gradientMem,
            activations = activationMem,
            kvCache = kvCacheMem,
            frameworkOverhead = frameworkMem,
            total = totalMem
        )
    }
    
    private fun calculateParameterMemory(modelConfig: ModelConfig, precision: PrecisionMode): MemoryEstimate {
        val bytesPerParam = when(precision) {
            PrecisionMode.FP32 -> 4.0
            PrecisionMode.FP16, PrecisionMode.BF16 -> 2.0
            PrecisionMode.INT8 -> 1.0
            PrecisionMode.INT4 -> 0.5
        }
        val baseMemoryGB = (modelConfig.totalParams * bytesPerParam * 1.2) / 1e9
        
        return MemoryEstimate(
            minimumGB = baseMemoryGB * 0.9,
            typicalGB = baseMemoryGB,
            maximumGB = baseMemoryGB * 1.1,
            safetyBufferGB = baseMemoryGB * 0.15,
            confidence = "high",
            notes = listOf("Includes 20% overhead")
        )
    }
    
    private fun calculateKVCacheMemory(
        modelConfig: ModelConfig,
        precision: PrecisionMode,
        operation: OperationMode,
        batchSize: Int,
        sequenceLength: Int
    ): MemoryEstimate {
        if (operation != OperationMode.INFERENCE) {
            return MemoryEstimate(0.0, 0.0, 0.0, 0.0, "high", listOf("KV cache not used"))
        }
        
        val bytesPerKVEntry = if (precision in listOf(PrecisionMode.INT8, PrecisionMode.INT4)) 1.0 else 2.0
        val kvCacheGB = (2.0 * batchSize * sequenceLength * (modelConfig.hiddenSize ?: 4096) * bytesPerKVEntry * (modelConfig.numHiddenLayers ?: 32) / 8) / 1e9
        
        return MemoryEstimate(
            minimumGB = kvCacheGB * 0.85,
            typicalGB = kvCacheGB,
            maximumGB = kvCacheGB * 1.15,
            safetyBufferGB = kvCacheGB * 0.2,
            confidence = "medium",
            notes = listOf("KV cache memory")
        )
    }
    
    private fun calculateActivationMemory(
        modelConfig: ModelConfig,
        precision: PrecisionMode,
        operation: OperationMode,
        batchSize: Int,
        sequenceLength: Int
    ): MemoryEstimate {
        val baseGB = if (operation == OperationMode.INFERENCE) {
            0.1 * batchSize * sequenceLength / 1024.0
        } else {
            8.0 * batchSize * sequenceLength / 1024.0
        }
        
        return MemoryEstimate(
            minimumGB = baseGB * 0.5,
            typicalGB = baseGB,
            maximumGB = baseGB * 1.5,
            safetyBufferGB = baseGB * 0.5,
            confidence = "low",
            notes = listOf("Activation memory")
        )
    }
    
    private fun calculateOptimizerMemory(modelConfig: ModelConfig, precision: PrecisionMode, operation: OperationMode): MemoryEstimate {
        if (operation == OperationMode.INFERENCE) {
            return MemoryEstimate(0.0, 0.0, 0.0, 0.0, "high", listOf("No optimizer for inference"))
        }
        val baseMemoryGB = (modelConfig.totalParams * 8) / 1e9
        return MemoryEstimate(
            minimumGB = baseMemoryGB * 0.8,
            typicalGB = baseMemoryGB,
            maximumGB = baseMemoryGB * 1.2,
            safetyBufferGB = baseMemoryGB * 0.3,
            confidence = "medium",
            notes = listOf("AdamW optimizer")
        )
    }
    
    private fun calculateGradientMemory(modelConfig: ModelConfig, precision: PrecisionMode, operation: OperationMode): MemoryEstimate {
        if (operation == OperationMode.INFERENCE) {
            return MemoryEstimate(0.0, 0.0, 0.0, 0.0, "high", listOf("No gradients for inference"))
        }
        val bytesPerParam = when(precision) {
            PrecisionMode.FP32 -> 4.0
            PrecisionMode.FP16, PrecisionMode.BF16 -> 2.0
            PrecisionMode.INT8 -> 1.0
            PrecisionMode.INT4 -> 0.5
        }
        val baseMemoryGB = (modelConfig.totalParams * bytesPerParam) / 1e9
        return MemoryEstimate(
            minimumGB = baseMemoryGB * 0.85,
            typicalGB = baseMemoryGB,
            maximumGB = baseMemoryGB * 1.15,
            safetyBufferGB = baseMemoryGB * 0.25,
            confidence = "medium",
            notes = listOf("Gradient memory")
        )
    }
    
    private fun calculateFrameworkOverhead(modelConfig: ModelConfig, precision: PrecisionMode, operation: OperationMode): MemoryEstimate {
        val baseGB = if (operation == OperationMode.INFERENCE) 1.0 else 2.0
        return MemoryEstimate(
            minimumGB = baseGB * 0.5,
            typicalGB = baseGB,
            maximumGB = baseGB * 2.0,
            safetyBufferGB = baseGB * 0.5,
            confidence = "low",
            notes = listOf("Framework overhead")
        )
    }
    
    private fun calculateTotalMemory(components: List<MemoryEstimate>): MemoryEstimate {
        val minTotal = components.sumOf { it.minimumGB }
        val typicalTotal = components.sumOf { it.typicalGB }
        val maxTotal = components.sumOf { it.maximumGB }
        val safetyBuffer = components.sumOf { it.safetyBufferGB } + (typicalTotal * 0.2)
        
        return MemoryEstimate(
            minimumGB = minTotal,
            typicalGB = typicalTotal,
            maximumGB = maxTotal,
            safetyBufferGB = safetyBuffer,
            confidence = "medium",
            notes = listOf("Total VRAM requirement")
        )
    }
    
    fun getRecommendedGPU(totalEstimate: MemoryEstimate): String {
        val safeRequirement = totalEstimate.maximumGB + totalEstimate.safetyBufferGB
        return when {
            safeRequirement < 8 -> "RTX 3060 (12GB)"
            safeRequirement < 16 -> "RTX 4080 (16GB)"
            safeRequirement < 24 -> "RTX 4090 (24GB)"
            safeRequirement < 48 -> "A6000 (48GB)"
            else -> "Multiple high-end GPUs"
        }
    }
}
