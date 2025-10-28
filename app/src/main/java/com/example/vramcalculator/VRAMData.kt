package com.example.vramcalculator

enum class PrecisionMode(val value: String) {
    FP32("fp32"), FP16("fp16"), BF16("bf16"), INT8("int8"), INT4("int4")
}

enum class OperationMode(val value: String) {
    INFERENCE("inference"), TRAINING("training"), FINE_TUNING("fine_tuning")
}

data class ModelConfig(
    val modelId: String,
    val totalParams: Long,
    val hiddenSize: Int? = null,
    val numHiddenLayers: Int? = null,
    val numAttentionHeads: Int? = null,
    val numKeyValueHeads: Int? = null,
    val intermediateSize: Int? = null,
    val vocabSize: Int? = null,
    val modelType: String? = null
)

data class MemoryEstimate(
    val minimumGB: Double,
    val typicalGB: Double,
    val maximumGB: Double,
    val safetyBufferGB: Double,
    val confidence: String,
    val notes: List<String>
)

data class VRAMBreakdown(
    val parameters: MemoryEstimate,
    val optimizer: MemoryEstimate,
    val gradients: MemoryEstimate,
    val activations: MemoryEstimate,
    val kvCache: MemoryEstimate,
    val frameworkOverhead: MemoryEstimate,
    val total: MemoryEstimate
)
