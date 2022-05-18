package main

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

var ignored = map[string]struct{}{
	// "cudnnActivationBackward":{}, //
	// "cudnnActivationForward":{}, //
	// "cudnnAddTensor":{}, //
	"cudnnAdvInferVersionCheck":     {},
	"cudnnAdvTrainVersionCheck":     {},
	"cudnnBackendCreateDescriptor":  {},
	"cudnnBackendDestroyDescriptor": {},
	"cudnnBackendExecute":           {},
	"cudnnBackendFinalize":          {},
	"cudnnBackendGetAttribute":      {},
	"cudnnBackendInitialize":        {},
	// "cudnnBackendSetAttribute":      {},
	// "cudnnBatchNormalizationBackward":{}, //
	"cudnnBatchNormalizationBackwardEx": {},
	// "cudnnBatchNormalizationForwardInference":{}, //
	// "cudnnBatchNormalizationForwardTraining":{}, //
	"cudnnBatchNormalizationForwardTrainingEx": {},
	"cudnnBuildRNNDynamic":                     {},
	// "cudnnCTCLoss":{}, //
	"cudnnCTCLoss_v8":           {},
	"cudnnCnnInferVersionCheck": {},
	"cudnnCnnTrainVersionCheck": {},
	// "cudnnConvolutionBackwardBias":{}, //
	// "cudnnConvolutionBackwardData":{}, //
	// "cudnnConvolutionBackwardFilter":{}, //
	// "cudnnConvolutionBiasActivationForward":{}, //
	// "cudnnConvolutionForward":{}, //
	"cudnnCopyAlgorithmDescriptor": {},
	"cudnnCreate":                  {},
	// "cudnnCreateActivationDescriptor":{}, //
	"cudnnCreateAlgorithmDescriptor":  {},
	"cudnnCreateAlgorithmPerformance": {},
	"cudnnCreateAttnDescriptor":       {},
	// "cudnnCreateCTCLossDescriptor":{}, //
	"cudnnCreateConvolutionDescriptor": {},
	// "cudnnCreateDropoutDescriptor":{}, //
	// "cudnnCreateFilterDescriptor":{}, //
	"cudnnCreateFusedOpsConstParamPack":   {},
	"cudnnCreateFusedOpsPlan":             {},
	"cudnnCreateFusedOpsVariantParamPack": {},
	// "cudnnCreateLRNDescriptor":{}, //
	// "cudnnCreateOpTensorDescriptor":{}, //
	// "cudnnCreatePersistentRNNPlan":{}, //
	// "cudnnCreatePoolingDescriptor":{}, //
	"cudnnCreateRNNDataDescriptor": {},
	// "cudnnCreateRNNDescriptor":{}, //
	// "cudnnCreateReduceTensorDescriptor":{}, //
	"cudnnCreateSeqDataDescriptor": {},
	// "cudnnCreateSpatialTransformerDescriptor":{}, //
	// "cudnnCreateTensorDescriptor":{}, //
	"cudnnCreateTensorTransformDescriptor": {},
	// "cudnnDeriveBNTensorDescriptor":{}, //
	"cudnnDeriveNormTensorDescriptor": {},
	"cudnnDestroy":                    {},
	// "cudnnDestroyActivationDescriptor":{}, //
	"cudnnDestroyAlgorithmDescriptor":  {},
	"cudnnDestroyAlgorithmPerformance": {},
	"cudnnDestroyAttnDescriptor":       {},
	// "cudnnDestroyCTCLossDescriptor":{}, //
	"cudnnDestroyConvolutionDescriptor": {},
	// "cudnnDestroyDropoutDescriptor":{}, //
	// "cudnnDestroyFilterDescriptor":{}, //
	"cudnnDestroyFusedOpsConstParamPack":   {},
	"cudnnDestroyFusedOpsPlan":             {},
	"cudnnDestroyFusedOpsVariantParamPack": {},
	// "cudnnDestroyLRNDescriptor":{}, //
	// "cudnnDestroyOpTensorDescriptor":{}, //
	// "cudnnDestroyPersistentRNNPlan":{}, //
	// "cudnnDestroyPoolingDescriptor":{}, //
	"cudnnDestroyRNNDataDescriptor": {},
	// "cudnnDestroyRNNDescriptor":{}, //
	// "cudnnDestroyReduceTensorDescriptor":{}, //
	"cudnnDestroySeqDataDescriptor": {},
	// "cudnnDestroySpatialTransformerDescriptor":{}, //
	// "cudnnDestroyTensorDescriptor":{}, //
	"cudnnDestroyTensorTransformDescriptor": {},
	// "cudnnDivisiveNormalizationBackward":{}, //
	// "cudnnDivisiveNormalizationForward":{}, //
	// "cudnnDropoutBackward":{}, //
	// "cudnnDropoutForward":{}, //
	// "cudnnDropoutGetReserveSpaceSize":{}, //
	// "cudnnDropoutGetStatesSize":{}, //
	// "cudnnFindConvolutionBackwardDataAlgorithm":{}, //
	// "cudnnFindConvolutionBackwardDataAlgorithmEx":{}, //
	// "cudnnFindConvolutionBackwardFilterAlgorithm":{}, //
	// "cudnnFindConvolutionBackwardFilterAlgorithmEx":{}, //
	// "cudnnFindConvolutionForwardAlgorithm":{}, //
	// "cudnnFindConvolutionForwardAlgorithmEx":{}, //
	"cudnnFindRNNBackwardDataAlgorithmEx":                      {},
	"cudnnFindRNNBackwardWeightsAlgorithmEx":                   {},
	"cudnnFindRNNForwardInferenceAlgorithmEx":                  {},
	"cudnnFindRNNForwardTrainingAlgorithmEx":                   {},
	"cudnnFusedOpsExecute":                                     {},
	"cudnnGetActivationDescriptor":                             {},
	"cudnnGetAlgorithmDescriptor":                              {},
	"cudnnGetAlgorithmPerformance":                             {},
	"cudnnGetAlgorithmSpaceSize":                               {},
	"cudnnGetAttnDescriptor":                                   {},
	"cudnnGetBatchNormalizationBackwardExWorkspaceSize":        {},
	"cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize": {},
	"cudnnGetBatchNormalizationTrainingExReserveSpaceSize":     {},
	"cudnnGetCTCLossDescriptor":                                {},
	"cudnnGetCTCLossDescriptorEx":                              {},
	"cudnnGetCTCLossDescriptor_v8":                             {},
	"cudnnGetCTCLossWorkspaceSize":                             {},
	"cudnnGetCTCLossWorkspaceSize_v8":                          {},
	"cudnnGetCallback":                                         {},
	"cudnnGetConvolution2dDescriptor":                          {},
	"cudnnGetConvolution2dForwardOutputDim":                    {},
	"cudnnGetConvolutionBackwardDataAlgorithmMaxCount":         {},
	"cudnnGetConvolutionBackwardDataAlgorithm_v7":              {},
	"cudnnGetConvolutionBackwardDataWorkspaceSize":             {},
	"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount":       {},
	"cudnnGetConvolutionBackwardFilterAlgorithm_v7":            {},
	"cudnnGetConvolutionBackwardFilterWorkspaceSize":           {},
	"cudnnGetConvolutionForwardAlgorithmMaxCount":              {},
	"cudnnGetConvolutionForwardAlgorithm_v7":                   {},
	"cudnnGetConvolutionForwardWorkspaceSize":                  {},
	"cudnnGetConvolutionGroupCount":                            {},
	"cudnnGetConvolutionMathType":                              {},
	"cudnnGetConvolutionNdDescriptor":                          {},
	"cudnnGetConvolutionNdForwardOutputDim":                    {},
	"cudnnGetConvolutionReorderType":                           {},
	"cudnnGetCudartVersion":                                    {},
	"cudnnGetDropoutDescriptor":                                {},
	"cudnnGetErrorString":                                      {},
	"cudnnGetFilter4dDescriptor":                               {},
	"cudnnGetFilterNdDescriptor":                               {},
	"cudnnGetFilterSizeInBytes":                                {},
	"cudnnGetFoldedConvBackwardDataDescriptors":                {},
	"cudnnGetFusedOpsConstParamPackAttribute":                  {},
	"cudnnGetFusedOpsVariantParamPackAttribute":                {},
	"cudnnGetLRNDescriptor":                                    {},
	"cudnnGetMultiHeadAttnBuffers":                             {},
	"cudnnGetMultiHeadAttnWeights":                             {},
	"cudnnGetNormalizationBackwardWorkspaceSize":               {},
	"cudnnGetNormalizationForwardTrainingWorkspaceSize":        {},
	"cudnnGetNormalizationTrainingReserveSpaceSize":            {},
	"cudnnGetOpTensorDescriptor":                               {},
	"cudnnGetPooling2dDescriptor":                              {},
	"cudnnGetPooling2dForwardOutputDim":                        {},
	"cudnnGetPoolingNdDescriptor":                              {},
	"cudnnGetPoolingNdForwardOutputDim":                        {},
	"cudnnGetProperty":                                         {},
	"cudnnGetRNNBackwardDataAlgorithmMaxCount":                 {},
	"cudnnGetRNNBackwardWeightsAlgorithmMaxCount":              {},
	"cudnnGetRNNBiasMode":                                      {},
	"cudnnGetRNNDataDescriptor":                                {},
	"cudnnGetRNNDescriptor_v6":                                 {},
	"cudnnGetRNNDescriptor_v8":                                 {},
	"cudnnGetRNNForwardInferenceAlgorithmMaxCount":             {},
	"cudnnGetRNNForwardTrainingAlgorithmMaxCount":              {},
	"cudnnGetRNNLinLayerBiasParams":                            {}, //
	"cudnnGetRNNLinLayerMatrixParams":                          {}, //
	"cudnnGetRNNMatrixMathType":                                {},
	"cudnnGetRNNPaddingMode":                                   {},
	// "cudnnGetRNNParamsSize":{}, //
	"cudnnGetRNNProjectionLayers": {},
	"cudnnGetRNNTempSpaceSizes":   {},
	// "cudnnGetRNNTrainingReserveSize":{}, //
	"cudnnGetRNNWeightParams":    {},
	"cudnnGetRNNWeightSpaceSize": {},
	// "cudnnGetRNNWorkspaceSize":{}, //
	"cudnnGetReduceTensorDescriptor": {},
	// "cudnnGetReductionIndicesSize":{}, //
	// "cudnnGetReductionWorkspaceSize":{}, //
	"cudnnGetSeqDataDescriptor":         {},
	"cudnnGetStream":                    {},
	"cudnnGetTensor4dDescriptor":        {},
	"cudnnGetTensorNdDescriptor":        {},
	"cudnnGetTensorSizeInBytes":         {},
	"cudnnGetTensorTransformDescriptor": {},
	"cudnnGetVersion":                   {},
	// "cudnnIm2Col":{}, //
	"cudnnInitTransformDest": {},
	// "cudnnLRNCrossChannelBackward":{}, //
	// "cudnnLRNCrossChannelForward":{}, //
	"cudnnMakeFusedOpsPlan":              {},
	"cudnnMultiHeadAttnBackwardData":     {},
	"cudnnMultiHeadAttnBackwardWeights":  {},
	"cudnnMultiHeadAttnForward":          {},
	"cudnnNormalizationBackward":         {},
	"cudnnNormalizationForwardInference": {},
	"cudnnNormalizationForwardTraining":  {},
	// "cudnnOpTensor":{}, //
	"cudnnOpsInferVersionCheck": {},
	"cudnnOpsTrainVersionCheck": {},
	// "cudnnPoolingBackward":{}, //
	// "cudnnPoolingForward":{}, //
	"cudnnQueryRuntimeError": {},
	// "cudnnRNNBackwardData":{}, //
	"cudnnRNNBackwardDataEx":  {},
	"cudnnRNNBackwardData_v8": {},
	// "cudnnRNNBackwardWeights":{}, //
	"cudnnRNNBackwardWeightsEx":  {},
	"cudnnRNNBackwardWeights_v8": {},
	"cudnnRNNForward":            {},
	// "cudnnRNNForwardInference":{}, //
	"cudnnRNNForwardInferenceEx": {},
	"cudnnRNNForwardTraining":    {}, // looks to be deprecated
	"cudnnRNNForwardTrainingEx":  {},
	"cudnnRNNGetClip":            {},
	"cudnnRNNGetClip_v8":         {},
	"cudnnRNNSetClip":            {},
	"cudnnRNNSetClip_v8":         {},
	// "cudnnReduceTensor":{}, //
	"cudnnReorderFilterAndBias": {},
	"cudnnRestoreAlgorithm":     {},
	// "cudnnRestoreDropoutDescriptor":{}, //
	"cudnnSaveAlgorithm": {},
	// "cudnnScaleTensor":{}, //
	// "cudnnSetActivationDescriptor":{}, //
	// "cudnnSetAlgorithmDescriptor":  {}, //
	// "cudnnSetAlgorithmPerformance": {}, //
	// "cudnnSetAttnDescriptor":       {}, //
	// "cudnnSetCTCLossDescriptor":{}, //
	//"cudnnSetCTCLossDescriptorEx":     {},//
	//"cudnnSetCTCLossDescriptor_v8":    {},//
	"cudnnSetCallback":                {},
	"cudnnSetConvolution2dDescriptor": {},
	"cudnnSetConvolutionGroupCount":   {},
	"cudnnSetConvolutionMathType":     {},
	"cudnnSetConvolutionNdDescriptor": {},
	"cudnnSetConvolutionReorderType":  {},
	// "cudnnSetDropoutDescriptor":{}, //
	// "cudnnSetFilter4dDescriptor":{}, //
	// "cudnnSetFilterNdDescriptor":{}, //
	// "cudnnSetFusedOpsConstParamPackAttribute":   {}, //
	// "cudnnSetFusedOpsVariantParamPackAttribute": {}, //
	// "cudnnSetLRNDescriptor":{}, //
	// "cudnnSetOpTensorDescriptor":{}, //
	// "cudnnSetPersistentRNNPlan":{}, //
	// "cudnnSetPooling2dDescriptor":{}, //
	// "cudnnSetPoolingNdDescriptor":{}, //
	"cudnnSetRNNAlgorithmDescriptor": {},
	"cudnnSetRNNBiasMode":            {},
	//"cudnnSetRNNDataDescriptor":      {}, //
	// "cudnnSetRNNDescriptor_v6":{}, //
	"cudnnSetRNNDescriptor_v8": {},
	// "cudnnSetRNNMatrixMathType":{}, //
	"cudnnSetRNNPaddingMode":      {},
	"cudnnSetRNNProjectionLayers": {},
	// "cudnnSetReduceTensorDescriptor":{}, //
	// "cudnnSetSeqDataDescriptor": {}, //
	// "cudnnSetSpatialTransformerNdDescriptor":{}, //
	// "cudnnSetStream": {},//
	// "cudnnSetTensor":{}, //
	// "cudnnSetTensor4dDescriptor":{}, //
	// "cudnnSetTensor4dDescriptorEx":{}, //
	// "cudnnSetTensorNdDescriptor":{}, //
	// "cudnnSetTensorNdDescriptorEx":{}, //
	// "cudnnSetTensorTransformDescriptor": {}, //
	// "cudnnSoftmaxBackward":{}, //
	// "cudnnSoftmaxForward":{}, //
	// "cudnnSpatialTfGridGeneratorBackward":{}, //
	// "cudnnSpatialTfGridGeneratorForward":{}, //
	// "cudnnSpatialTfSamplerBackward":{}, //
	// "cudnnSpatialTfSamplerForward":{}, //
	"cudnnTransformFilter": {},
	// "cudnnTransformTensor":{}, //
	"cudnnTransformTensorEx": {},
}

func init() {

	fnNameMap = map[string]string{
		"cudnnActivationBackward":                                  "ActivationBackward",
		"cudnnActivationForward":                                   "ActivationForward",
		"cudnnAddTensor":                                           "AddTensor",
		"cudnnAdvInferVersionCheck":                                "AdvInferVersionCheck",
		"cudnnAdvTrainVersionCheck":                                "AdvTrainVersionCheck",
		"cudnnBackendCreateDescriptor":                             "BackendCreateDescriptor",
		"cudnnBackendDestroyDescriptor":                            "BackendDestroyDescriptor",
		"cudnnBackendExecute":                                      "BackendExecute",
		"cudnnBackendFinalize":                                     "BackendFinalize",
		"cudnnBackendGetAttribute":                                 "BackendGetAttribute",
		"cudnnBackendInitialize":                                   "BackendInitialize",
		"cudnnBackendSetAttribute":                                 "BackendSetAttribute",
		"cudnnBatchNormalizationBackward":                          "BatchNormalizationBackward",
		"cudnnBatchNormalizationBackwardEx":                        "BatchNormalizationBackwardEx",
		"cudnnBatchNormalizationForwardInference":                  "BatchNormalizationForwardInference",
		"cudnnBatchNormalizationForwardTraining":                   "BatchNormalizationForwardTraining",
		"cudnnBatchNormalizationForwardTrainingEx":                 "BatchNormalizationForwardTrainingEx",
		"cudnnBuildRNNDynamic":                                     "BuildRNNDynamic",
		"cudnnCTCLoss":                                             "CTCLoss",
		"cudnnCTCLoss_v8":                                          "CTCLoss_v8",
		"cudnnCnnInferVersionCheck":                                "CnnInferVersionCheck",
		"cudnnCnnTrainVersionCheck":                                "CnnTrainVersionCheck",
		"cudnnConvolutionBackwardBias":                             "ConvolutionBackwardBias",
		"cudnnConvolutionBackwardData":                             "ConvolutionBackwardData",
		"cudnnConvolutionBackwardFilter":                           "ConvolutionBackwardFilter",
		"cudnnConvolutionBiasActivationForward":                    "ConvolutionBiasActivationForward",
		"cudnnConvolutionForward":                                  "ConvolutionForward",
		"cudnnCopyAlgorithmDescriptor":                             "CopyAlgorithmDescriptor",
		"cudnnCreate":                                              "Create",
		"cudnnCreateActivationDescriptor":                          "CreateActivationDescriptor",
		"cudnnCreateAlgorithmDescriptor":                           "CreateAlgorithmDescriptor",
		"cudnnCreateAlgorithmPerformance":                          "CreateAlgorithmPerformance",
		"cudnnCreateAttnDescriptor":                                "CreateAttnDescriptor",
		"cudnnCreateCTCLossDescriptor":                             "CreateCTCLossDescriptor",
		"cudnnCreateConvolutionDescriptor":                         "CreateConvolutionDescriptor",
		"cudnnCreateDropoutDescriptor":                             "CreateDropoutDescriptor",
		"cudnnCreateFilterDescriptor":                              "CreateFilterDescriptor",
		"cudnnCreateFusedOpsConstParamPack":                        "CreateFusedOpsConstParamPack",
		"cudnnCreateFusedOpsPlan":                                  "CreateFusedOpsPlan",
		"cudnnCreateFusedOpsVariantParamPack":                      "CreateFusedOpsVariantParamPack",
		"cudnnCreateLRNDescriptor":                                 "CreateLRNDescriptor",
		"cudnnCreateOpTensorDescriptor":                            "CreateOpTensorDescriptor",
		"cudnnCreatePersistentRNNPlan":                             "CreatePersistentRNNPlan",
		"cudnnCreatePoolingDescriptor":                             "CreatePoolingDescriptor",
		"cudnnCreateRNNDataDescriptor":                             "CreateRNNDataDescriptor",
		"cudnnCreateRNNDescriptor":                                 "CreateRNNDescriptor",
		"cudnnCreateReduceTensorDescriptor":                        "CreateReduceTensorDescriptor",
		"cudnnCreateSeqDataDescriptor":                             "CreateSeqDataDescriptor",
		"cudnnCreateSpatialTransformerDescriptor":                  "CreateSpatialTransformerDescriptor",
		"cudnnCreateTensorDescriptor":                              "CreateTensorDescriptor",
		"cudnnCreateTensorTransformDescriptor":                     "CreateTensorTransformDescriptor",
		"cudnnDeriveBNTensorDescriptor":                            "DeriveBNTensorDescriptor",
		"cudnnDeriveNormTensorDescriptor":                          "DeriveNormTensorDescriptor",
		"cudnnDestroy":                                             "Destroy",
		"cudnnDestroyActivationDescriptor":                         "DestroyActivationDescriptor",
		"cudnnDestroyAlgorithmDescriptor":                          "DestroyAlgorithmDescriptor",
		"cudnnDestroyAlgorithmPerformance":                         "DestroyAlgorithmPerformance",
		"cudnnDestroyAttnDescriptor":                               "DestroyAttnDescriptor",
		"cudnnDestroyCTCLossDescriptor":                            "DestroyCTCLossDescriptor",
		"cudnnDestroyConvolutionDescriptor":                        "DestroyConvolutionDescriptor",
		"cudnnDestroyDropoutDescriptor":                            "DestroyDropoutDescriptor",
		"cudnnDestroyFilterDescriptor":                             "DestroyFilterDescriptor",
		"cudnnDestroyFusedOpsConstParamPack":                       "DestroyFusedOpsConstParamPack",
		"cudnnDestroyFusedOpsPlan":                                 "DestroyFusedOpsPlan",
		"cudnnDestroyFusedOpsVariantParamPack":                     "DestroyFusedOpsVariantParamPack",
		"cudnnDestroyLRNDescriptor":                                "DestroyLRNDescriptor",
		"cudnnDestroyOpTensorDescriptor":                           "DestroyOpTensorDescriptor",
		"cudnnDestroyPersistentRNNPlan":                            "DestroyPersistentRNNPlan",
		"cudnnDestroyPoolingDescriptor":                            "DestroyPoolingDescriptor",
		"cudnnDestroyRNNDataDescriptor":                            "DestroyRNNDataDescriptor",
		"cudnnDestroyRNNDescriptor":                                "DestroyRNNDescriptor",
		"cudnnDestroyReduceTensorDescriptor":                       "DestroyReduceTensorDescriptor",
		"cudnnDestroySeqDataDescriptor":                            "DestroySeqDataDescriptor",
		"cudnnDestroySpatialTransformerDescriptor":                 "DestroySpatialTransformerDescriptor",
		"cudnnDestroyTensorDescriptor":                             "DestroyTensorDescriptor",
		"cudnnDestroyTensorTransformDescriptor":                    "DestroyTensorTransformDescriptor",
		"cudnnDivisiveNormalizationBackward":                       "DivisiveNormalizationBackward",
		"cudnnDivisiveNormalizationForward":                        "DivisiveNormalizationForward",
		"cudnnDropoutBackward":                                     "DropoutBackward",
		"cudnnDropoutForward":                                      "DropoutForward",
		"cudnnDropoutGetReserveSpaceSize":                          "DropoutGetReserveSpaceSize",
		"cudnnDropoutGetStatesSize":                                "DropoutGetStatesSize",
		"cudnnFindConvolutionBackwardDataAlgorithm":                "FindConvolutionBackwardDataAlgorithm",
		"cudnnFindConvolutionBackwardDataAlgorithmEx":              "FindConvolutionBackwardDataAlgorithmEx",
		"cudnnFindConvolutionBackwardFilterAlgorithm":              "FindConvolutionBackwardFilterAlgorithm",
		"cudnnFindConvolutionBackwardFilterAlgorithmEx":            "FindConvolutionBackwardFilterAlgorithmEx",
		"cudnnFindConvolutionForwardAlgorithm":                     "FindConvolutionForwardAlgorithm",
		"cudnnFindConvolutionForwardAlgorithmEx":                   "FindConvolutionForwardAlgorithmEx",
		"cudnnFindRNNBackwardDataAlgorithmEx":                      "FindRNNBackwardDataAlgorithmEx",
		"cudnnFindRNNBackwardWeightsAlgorithmEx":                   "FindRNNBackwardWeightsAlgorithmEx",
		"cudnnFindRNNForwardInferenceAlgorithmEx":                  "FindRNNForwardInferenceAlgorithmEx",
		"cudnnFindRNNForwardTrainingAlgorithmEx":                   "FindRNNForwardTrainingAlgorithmEx",
		"cudnnFusedOpsExecute":                                     "FusedOpsExecute",
		"cudnnGetActivationDescriptor":                             "GetActivationDescriptor",
		"cudnnGetAlgorithmDescriptor":                              "GetAlgorithmDescriptor",
		"cudnnGetAlgorithmPerformance":                             "GetAlgorithmPerformance",
		"cudnnGetAlgorithmSpaceSize":                               "GetAlgorithmSpaceSize",
		"cudnnGetAttnDescriptor":                                   "GetAttnDescriptor",
		"cudnnGetBatchNormalizationBackwardExWorkspaceSize":        "GetBatchNormalizationBackwardExWorkspaceSize",
		"cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize": "GetBatchNormalizationForwardTrainingExWorkspaceSize",
		"cudnnGetBatchNormalizationTrainingExReserveSpaceSize":     "GetBatchNormalizationTrainingExReserveSpaceSize",
		"cudnnGetCTCLossDescriptor":                                "GetCTCLossDescriptor",
		"cudnnGetCTCLossDescriptorEx":                              "GetCTCLossDescriptorEx",
		"cudnnGetCTCLossDescriptor_v8":                             "GetCTCLossDescriptor_v8",
		"cudnnGetCTCLossWorkspaceSize":                             "GetCTCLossWorkspaceSize",
		"cudnnGetCTCLossWorkspaceSize_v8":                          "GetCTCLossWorkspaceSize_v8",
		"cudnnGetCallback":                                         "GetCallback",
		"cudnnGetConvolution2dDescriptor":                          "GetConvolution2dDescriptor",
		"cudnnGetConvolution2dForwardOutputDim":                    "GetConvolution2dForwardOutputDim",
		"cudnnGetConvolutionBackwardDataAlgorithmMaxCount":         "GetConvolutionBackwardDataAlgorithmMaxCount",
		"cudnnGetConvolutionBackwardDataAlgorithm_v7":              "GetConvolutionBackwardDataAlgorithm_v7",
		"cudnnGetConvolutionBackwardDataWorkspaceSize":             "GetConvolutionBackwardDataWorkspaceSize",
		"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount":       "GetConvolutionBackwardFilterAlgorithmMaxCount",
		"cudnnGetConvolutionBackwardFilterAlgorithm_v7":            "GetConvolutionBackwardFilterAlgorithm_v7",
		"cudnnGetConvolutionBackwardFilterWorkspaceSize":           "GetConvolutionBackwardFilterWorkspaceSize",
		"cudnnGetConvolutionForwardAlgorithmMaxCount":              "GetConvolutionForwardAlgorithmMaxCount",
		"cudnnGetConvolutionForwardAlgorithm_v7":                   "GetConvolutionForwardAlgorithm_v7",
		"cudnnGetConvolutionForwardWorkspaceSize":                  "GetConvolutionForwardWorkspaceSize",
		"cudnnGetConvolutionGroupCount":                            "GetConvolutionGroupCount",
		"cudnnGetConvolutionMathType":                              "GetConvolutionMathType",
		"cudnnGetConvolutionNdDescriptor":                          "GetConvolutionNdDescriptor",
		"cudnnGetConvolutionNdForwardOutputDim":                    "GetConvolutionNdForwardOutputDim",
		"cudnnGetConvolutionReorderType":                           "GetConvolutionReorderType",
		"cudnnGetCudartVersion":                                    "GetCudartVersion",
		"cudnnGetDropoutDescriptor":                                "GetDropoutDescriptor",
		"cudnnGetErrorString":                                      "GetErrorString",
		"cudnnGetFilter4dDescriptor":                               "GetFilter4dDescriptor",
		"cudnnGetFilterNdDescriptor":                               "GetFilterNdDescriptor",
		"cudnnGetFilterSizeInBytes":                                "GetFilterSizeInBytes",
		"cudnnGetFoldedConvBackwardDataDescriptors":                "GetFoldedConvBackwardDataDescriptors",
		"cudnnGetFusedOpsConstParamPackAttribute":                  "GetFusedOpsConstParamPackAttribute",
		"cudnnGetFusedOpsVariantParamPackAttribute":                "GetFusedOpsVariantParamPackAttribute",
		"cudnnGetLRNDescriptor":                                    "GetLRNDescriptor",
		"cudnnGetMultiHeadAttnBuffers":                             "GetMultiHeadAttnBuffers",
		"cudnnGetMultiHeadAttnWeights":                             "GetMultiHeadAttnWeights",
		"cudnnGetNormalizationBackwardWorkspaceSize":               "GetNormalizationBackwardWorkspaceSize",
		"cudnnGetNormalizationForwardTrainingWorkspaceSize":        "GetNormalizationForwardTrainingWorkspaceSize",
		"cudnnGetNormalizationTrainingReserveSpaceSize":            "GetNormalizationTrainingReserveSpaceSize",
		"cudnnGetOpTensorDescriptor":                               "GetOpTensorDescriptor",
		"cudnnGetPooling2dDescriptor":                              "GetPooling2dDescriptor",
		"cudnnGetPooling2dForwardOutputDim":                        "GetPooling2dForwardOutputDim",
		"cudnnGetPoolingNdDescriptor":                              "GetPoolingNdDescriptor",
		"cudnnGetPoolingNdForwardOutputDim":                        "GetPoolingNdForwardOutputDim",
		"cudnnGetProperty":                                         "GetProperty",
		"cudnnGetRNNBackwardDataAlgorithmMaxCount":                 "GetRNNBackwardDataAlgorithmMaxCount",
		"cudnnGetRNNBackwardWeightsAlgorithmMaxCount":              "GetRNNBackwardWeightsAlgorithmMaxCount",
		"cudnnGetRNNBiasMode":                                      "GetRNNBiasMode",
		"cudnnGetRNNDataDescriptor":                                "GetRNNDataDescriptor",
		"cudnnGetRNNDescriptor_v6":                                 "GetRNNDescriptor_v6",
		"cudnnGetRNNDescriptor_v8":                                 "GetRNNDescriptor_v8",
		"cudnnGetRNNForwardInferenceAlgorithmMaxCount":             "GetRNNForwardInferenceAlgorithmMaxCount",
		"cudnnGetRNNForwardTrainingAlgorithmMaxCount":              "GetRNNForwardTrainingAlgorithmMaxCount",
		"cudnnGetRNNLinLayerBiasParams":                            "GetRNNLinLayerBiasParams",
		"cudnnGetRNNLinLayerMatrixParams":                          "GetRNNLinLayerMatrixParams",
		"cudnnGetRNNMatrixMathType":                                "GetRNNMatrixMathType",
		"cudnnGetRNNPaddingMode":                                   "GetRNNPaddingMode",
		"cudnnGetRNNParamsSize":                                    "GetRNNParamsSize",
		"cudnnGetRNNProjectionLayers":                              "GetRNNProjectionLayers",
		"cudnnGetRNNTempSpaceSizes":                                "GetRNNTempSpaceSizes",
		"cudnnGetRNNTrainingReserveSize":                           "GetRNNTrainingReserveSize",
		"cudnnGetRNNWeightParams":                                  "GetRNNWeightParams",
		"cudnnGetRNNWeightSpaceSize":                               "GetRNNWeightSpaceSize",
		"cudnnGetRNNWorkspaceSize":                                 "GetRNNWorkspaceSize",
		"cudnnGetReduceTensorDescriptor":                           "GetReduceTensorDescriptor",
		"cudnnGetReductionIndicesSize":                             "GetReductionIndicesSize",
		"cudnnGetReductionWorkspaceSize":                           "GetReductionWorkspaceSize",
		"cudnnGetSeqDataDescriptor":                                "GetSeqDataDescriptor",
		"cudnnGetStream":                                           "GetStream",
		"cudnnGetTensor4dDescriptor":                               "GetTensor4dDescriptor",
		"cudnnGetTensorNdDescriptor":                               "GetTensorNdDescriptor",
		"cudnnGetTensorSizeInBytes":                                "GetTensorSizeInBytes",
		"cudnnGetTensorTransformDescriptor":                        "GetTensorTransformDescriptor",
		"cudnnGetVersion":                                          "GetVersion",
		"cudnnIm2Col":                                              "Im2Col",
		"cudnnInitTransformDest":                                   "InitTransformDest",
		"cudnnLRNCrossChannelBackward":                             "LRNCrossChannelBackward",
		"cudnnLRNCrossChannelForward":                              "LRNCrossChannelForward",
		"cudnnMakeFusedOpsPlan":                                    "MakeFusedOpsPlan",
		"cudnnMultiHeadAttnBackwardData":                           "MultiHeadAttnBackwardData",
		"cudnnMultiHeadAttnBackwardWeights":                        "MultiHeadAttnBackwardWeights",
		"cudnnMultiHeadAttnForward":                                "MultiHeadAttnForward",
		"cudnnNormalizationBackward":                               "NormalizationBackward",
		"cudnnNormalizationForwardInference":                       "NormalizationForwardInference",
		"cudnnNormalizationForwardTraining":                        "NormalizationForwardTraining",
		"cudnnOpTensor":                                            "OpTensor",
		"cudnnOpsInferVersionCheck":                                "OpsInferVersionCheck",
		"cudnnOpsTrainVersionCheck":                                "OpsTrainVersionCheck",
		"cudnnPoolingBackward":                                     "PoolingBackward",
		"cudnnPoolingForward":                                      "PoolingForward",
		"cudnnQueryRuntimeError":                                   "QueryRuntimeError",
		"cudnnRNNBackwardData":                                     "RNNBackwardData",
		"cudnnRNNBackwardDataEx":                                   "RNNBackwardDataEx",
		"cudnnRNNBackwardData_v8":                                  "RNNBackwardData_v8",
		"cudnnRNNBackwardWeights":                                  "RNNBackwardWeights",
		"cudnnRNNBackwardWeightsEx":                                "RNNBackwardWeightsEx",
		"cudnnRNNBackwardWeights_v8":                               "RNNBackwardWeights_v8",
		"cudnnRNNForward":                                          "RNNForward",
		"cudnnRNNForwardInference":                                 "RNNForwardInference",
		"cudnnRNNForwardInferenceEx":                               "RNNForwardInferenceEx",
		"cudnnRNNForwardTraining":                                  "RNNForwardTraining",
		"cudnnRNNForwardTrainingEx":                                "RNNForwardTrainingEx",
		"cudnnRNNGetClip":                                          "RNNGetClip",
		"cudnnRNNGetClip_v8":                                       "RNNGetClip_v8",
		"cudnnRNNSetClip":                                          "RNNSetClip",
		"cudnnRNNSetClip_v8":                                       "RNNSetClip_v8",
		"cudnnReduceTensor":                                        "ReduceTensor",
		"cudnnReorderFilterAndBias":                                "ReorderFilterAndBias",
		"cudnnRestoreAlgorithm":                                    "RestoreAlgorithm",
		"cudnnRestoreDropoutDescriptor":                            "RestoreDropoutDescriptor",
		"cudnnSaveAlgorithm":                                       "SaveAlgorithm",
		"cudnnScaleTensor":                                         "ScaleTensor",
		"cudnnSetActivationDescriptor":                             "SetActivationDescriptor",
		"cudnnSetAlgorithmDescriptor":                              "SetAlgorithmDescriptor",
		"cudnnSetAlgorithmPerformance":                             "SetAlgorithmPerformance",
		"cudnnSetAttnDescriptor":                                   "SetAttnDescriptor",
		"cudnnSetCTCLossDescriptor":                                "SetCTCLossDescriptor",
		"cudnnSetCTCLossDescriptorEx":                              "SetCTCLossDescriptorEx",
		"cudnnSetCTCLossDescriptor_v8":                             "SetCTCLossDescriptor_v8",
		"cudnnSetCallback":                                         "SetCallback",
		"cudnnSetConvolution2dDescriptor":                          "SetConvolution2dDescriptor",
		"cudnnSetConvolutionGroupCount":                            "SetConvolutionGroupCount",
		"cudnnSetConvolutionMathType":                              "SetConvolutionMathType",
		"cudnnSetConvolutionNdDescriptor":                          "SetConvolutionNdDescriptor",
		"cudnnSetConvolutionReorderType":                           "SetConvolutionReorderType",
		"cudnnSetDropoutDescriptor":                                "SetDropoutDescriptor",
		"cudnnSetFilter4dDescriptor":                               "SetFilter4dDescriptor",
		"cudnnSetFilterNdDescriptor":                               "SetFilterNdDescriptor",
		"cudnnSetFusedOpsConstParamPackAttribute":                  "SetFusedOpsConstParamPackAttribute",
		"cudnnSetFusedOpsVariantParamPackAttribute":                "SetFusedOpsVariantParamPackAttribute",
		"cudnnSetLRNDescriptor":                                    "SetLRNDescriptor",
		"cudnnSetOpTensorDescriptor":                               "SetOpTensorDescriptor",
		"cudnnSetPersistentRNNPlan":                                "SetPersistentRNNPlan",
		"cudnnSetPooling2dDescriptor":                              "SetPooling2dDescriptor",
		"cudnnSetPoolingNdDescriptor":                              "SetPoolingNdDescriptor",
		"cudnnSetRNNAlgorithmDescriptor":                           "SetRNNAlgorithmDescriptor",
		"cudnnSetRNNBiasMode":                                      "SetRNNBiasMode",
		"cudnnSetRNNDataDescriptor":                                "SetRNNDataDescriptor",
		"cudnnSetRNNDescriptor_v6":                                 "SetRNNDescriptor_v6",
		"cudnnSetRNNDescriptor_v8":                                 "SetRNNDescriptor_v8",
		"cudnnSetRNNMatrixMathType":                                "SetRNNMatrixMathType",
		"cudnnSetRNNPaddingMode":                                   "SetRNNPaddingMode",
		"cudnnSetRNNProjectionLayers":                              "SetRNNProjectionLayers",
		"cudnnSetReduceTensorDescriptor":                           "SetReduceTensorDescriptor",
		"cudnnSetSeqDataDescriptor":                                "SetSeqDataDescriptor",
		"cudnnSetSpatialTransformerNdDescriptor":                   "SetSpatialTransformerNdDescriptor",
		"cudnnSetStream":                                           "SetStream",
		"cudnnSetTensor":                                           "SetTensor",
		"cudnnSetTensor4dDescriptor":                               "SetTensor4dDescriptor",
		"cudnnSetTensor4dDescriptorEx":                             "SetTensor4dDescriptorEx",
		"cudnnSetTensorNdDescriptor":                               "SetTensorNdDescriptor",
		"cudnnSetTensorNdDescriptorEx":                             "SetTensorNdDescriptorEx",
		"cudnnSetTensorTransformDescriptor":                        "SetTensorTransformDescriptor",
		"cudnnSoftmaxBackward":                                     "SoftmaxBackward",
		"cudnnSoftmaxForward":                                      "SoftmaxForward",
		"cudnnSpatialTfGridGeneratorBackward":                      "SpatialTfGridGeneratorBackward",
		"cudnnSpatialTfGridGeneratorForward":                       "SpatialTfGridGeneratorForward",
		"cudnnSpatialTfSamplerBackward":                            "SpatialTfSamplerBackward",
		"cudnnSpatialTfSamplerForward":                             "SpatialTfSamplerForward",
		"cudnnTransformFilter":                                     "TransformFilter",
		"cudnnTransformTensor":                                     "TransformTensor",
		"cudnnTransformTensorEx":                                   "TransformTensorEx",
	}
	enumMappings = map[string]string{
		"cudnnActivationMode_t":             "ActivationMode",
		"cudnnBackendAttributeName_t":       "BackendAttributeName",
		"cudnnBackendAttributeType_t":       "BackendAttributeType",
		"cudnnBackendDescriptorType_t":      "BackendDescriptorType",
		"cudnnBackendHeurMode_t":            "BackendHeurMode",
		"cudnnBackendKnobType_t":            "BackendKnobType",
		"cudnnBackendLayoutType_t":          "BackendLayoutType",
		"cudnnBackendNumericalNote_t":       "BackendNumericalNote",
		"cudnnBatchNormMode_t":              "BatchNormMode",
		"cudnnBatchNormOps_t":               "BatchNormOps",
		"cudnnCTCLossAlgo_t":                "CTCLossAlgo",
		"cudnnConvolutionBwdDataAlgo_t":     "ConvolutionBwdDataAlgo",
		"cudnnConvolutionBwdFilterAlgo_t":   "ConvolutionBwdFilterAlgo",
		"cudnnConvolutionFwdAlgo_t":         "ConvolutionFwdAlgo",
		"cudnnConvolutionMode_t":            "ConvolutionMode",
		"cudnnDataType_t":                   "DataType",
		"cudnnDeterminism_t":                "Determinism",
		"cudnnDirectionMode_t":              "DirectionMode",
		"cudnnDivNormMode_t":                "DivNormMode",
		"cudnnErrQueryMode_t":               "ErrQueryMode",
		"cudnnFoldingDirection_t":           "FoldingDirection",
		"cudnnForwardMode_t":                "ForwardMode",
		"cudnnFusedOpsConstParamLabel_t":    "FusedOpsConstParamLabel",
		"cudnnFusedOpsPointerPlaceHolder_t": "FusedOpsPointerPlaceHolder",
		"cudnnFusedOpsVariantParamLabel_t":  "FusedOpsVariantParamLabel",
		"cudnnFusedOps_t":                   "FusedOps",
		"cudnnGenStatsMode_t":               "GenStatsMode",
		"cudnnIndicesType_t":                "IndicesType",
		"cudnnLRNMode_t":                    "LRNMode",
		"cudnnLossNormalizationMode_t":      "LossNormalizationMode",
		"cudnnMathType_t":                   "MathType",
		"cudnnMultiHeadAttnWeightKind_t":    "MultiHeadAttnWeightKind",
		"cudnnNanPropagation_t":             "NanPropagation",
		"cudnnNormAlgo_t":                   "NormAlgo",
		"cudnnNormMode_t":                   "NormMode",
		"cudnnNormOps_t":                    "NormOps",
		"cudnnOpTensorOp_t":                 "OpTensorOp",
		"cudnnPointwiseMode_t":              "PointwiseMode",
		"cudnnPoolingMode_t":                "PoolingMode",
		"cudnnRNNAlgo_t":                    "RNNAlgo",
		"cudnnRNNBiasMode_t":                "RNNBiasMode",
		"cudnnRNNClipMode_t":                "RNNClipMode",
		"cudnnRNNDataLayout_t":              "RNNDataLayout",
		"cudnnRNNInputMode_t":               "RNNInputMode",
		"cudnnRNNMode_t":                    "RNNMode",
		"cudnnReduceTensorIndices_t":        "ReduceTensorIndices",
		"cudnnReduceTensorOp_t":             "ReduceTensorOp",
		"cudnnReorderType_t":                "ReorderType",
		"cudnnSamplerType_t":                "SamplerType",
		"cudnnSeqDataAxis_t":                "SeqDataAxis",
		"cudnnSeverity_t":                   "Severity",
		"cudnnSoftmaxAlgorithm_t":           "SoftmaxAlgorithm",
		"cudnnSoftmaxMode_t":                "SoftmaxMode",
		"cudnnStatus_t":                     "Status",
		"cudnnTensorFormat_t":               "TensorFormat",
		"cudnnWgradMode_t":                  "WgradMode",
	}
	alphaBetas = map[string]map[int]string{
		"cudnnActivationBackward":                  {9: "beta", 2: "alpha"},
		"cudnnActivationForward":                   {5: "beta", 2: "alpha"},
		"cudnnAddTensor":                           {4: "beta", 1: "alpha"},
		"cudnnBatchNormalizationBackward":          {5: "betaParamDiff", 4: "alphaParamDiff", 3: "betaDataDiff", 2: "alphaDataDiff"},
		"cudnnBatchNormalizationBackwardEx":        {6: "betaParamDiff", 5: "alphaParamDiff", 4: "betaDataDiff", 3: "alphaDataDiff"},
		"cudnnBatchNormalizationForwardInference":  {3: "beta", 2: "alpha"},
		"cudnnBatchNormalizationForwardTraining":   {3: "beta", 2: "alpha"},
		"cudnnBatchNormalizationForwardTrainingEx": {4: "beta", 3: "alpha"},
		"cudnnConvolutionBackwardBias":             {4: "beta", 1: "alpha"},
		"cudnnConvolutionBackwardData":             {10: "beta", 1: "alpha"},
		"cudnnConvolutionBackwardFilter":           {10: "beta", 1: "alpha"},
		"cudnnConvolutionBiasActivationForward":    {10: "alpha2", 1: "alpha1"},
		"cudnnConvolutionForward":                  {10: "beta", 1: "alpha"},
		"cudnnDivisiveNormalizationBackward":       {10: "beta", 3: "alpha"},
		"cudnnDivisiveNormalizationForward":        {9: "beta", 3: "alpha"},
		"cudnnLRNCrossChannelBackward":             {10: "beta", 3: "alpha"},
		"cudnnLRNCrossChannelForward":              {6: "beta", 3: "alpha"},
		"cudnnNormalizationBackward":               {7: "betaParamDiff", 6: "alphaParamDiff", 5: "betaDataDiff", 4: "alphaDataDiff"},
		"cudnnNormalizationForwardInference":       {5: "beta", 4: "alpha"},
		"cudnnNormalizationForwardTraining":        {5: "beta", 4: "alpha"},
		"cudnnOpTensor":                            {8: "beta", 5: "alpha2", 2: "alpha1"},
		"cudnnPoolingBackward":                     {9: "beta", 2: "alpha"},
		"cudnnPoolingForward":                      {5: "beta", 2: "alpha"},
		"cudnnReduceTensor":                        {9: "beta", 6: "alpha"},
		"cudnnScaleTensor":                         {3: "alpha"},
		"cudnnSoftmaxBackward":                     {8: "beta", 3: "alpha"},
		"cudnnSoftmaxForward":                      {6: "beta", 3: "alpha"},
		"cudnnSpatialTfSamplerBackward":            {5: "beta", 2: "alpha"},
		"cudnnSpatialTfSamplerForward":             {6: "beta", 2: "alpha"},
		"cudnnTransformFilter":                     {5: "beta", 2: "alpha"},
		"cudnnTransformTensor":                     {4: "beta", 1: "alpha"},
		"cudnnTransformTensorEx":                   {5: "beta", 2: "alpha"},
	}
	creations = map[string][]string{
		"cudnnActivationDescriptor_t":         {"cudnnCreateActivationDescriptor"},
		"cudnnAlgorithmDescriptor_t":          {"cudnnCreateAlgorithmDescriptor"},
		"cudnnAlgorithmPerformance_t":         {"cudnnCreateAlgorithmPerformance"},
		"cudnnAttnDescriptor_t":               {"cudnnCreateAttnDescriptor"},
		"cudnnBackendDescriptor_t":            {"cudnnBackendCreateDescriptor"},
		"cudnnConvolutionDescriptor_t":        {"cudnnCreateConvolutionDescriptor"},
		"cudnnDropoutDescriptor_t":            {"cudnnCreateDropoutDescriptor"},
		"cudnnFilterDescriptor_t":             {"cudnnCreateFilterDescriptor"},
		"cudnnFusedOpsConstParamPack_t":       {"cudnnCreateFusedOpsConstParamPack"},
		"cudnnFusedOpsPlan_t":                 {"cudnnCreateFusedOpsPlan"},
		"cudnnFusedOpsVariantParamPack_t":     {"cudnnCreateFusedOpsVariantParamPack"},
		"cudnnHandle_t":                       {"cudnnCreate"},
		"cudnnLRNDescriptor_t":                {"cudnnCreateLRNDescriptor"},
		"cudnnOpTensorDescriptor_t":           {"cudnnCreateOpTensorDescriptor"},
		"cudnnPersistentRNNPlan_t":            {"cudnnCreatePersistentRNNPlan"},
		"cudnnPoolingDescriptor_t":            {"cudnnCreatePoolingDescriptor"},
		"cudnnRNNDataDescriptor_t":            {"cudnnCreateRNNDataDescriptor"},
		"cudnnRNNDescriptor_t":                {"cudnnCreateRNNDescriptor"},
		"cudnnReduceTensorDescriptor_t":       {"cudnnCreateReduceTensorDescriptor"},
		"cudnnSeqDataDescriptor_t":            {"cudnnCreateSeqDataDescriptor"},
		"cudnnSpatialTransformerDescriptor_t": {"cudnnCreateSpatialTransformerDescriptor"},
		"cudnnTensorDescriptor_t":             {"cudnnCreateTensorDescriptor"},
		"cudnnTensorTransformDescriptor_t":    {"cudnnCreateTensorTransformDescriptor"},
		"cudnnCTCLossDescriptor_t":            {"cudnnCreateCTCLossDescriptor"},
	}

	setFns = map[string][]string{
		"cudaStream_t":                        {"cudnnSetStream"},
		"cudnnActivationDescriptor_t":         {"cudnnSetActivationDescriptor"},
		"cudnnAlgorithmDescriptor_t":          {"cudnnSetAlgorithmDescriptor"},
		"cudnnAlgorithmPerformance_t":         {"cudnnSetAlgorithmPerformance"},
		"cudnnAttnDescriptor_t":               {"cudnnSetAttnDescriptor"},
		"cudnnBackendDescriptor_t":            {"cudnnBackendSetAttribute"},
		"cudnnCTCLossDescriptor_t":            {"cudnnSetCTCLossDescriptor", "cudnnSetCTCLossDescriptorEx", "cudnnSetCTCLossDescriptor_v8"},
		"cudnnConvolutionDescriptor_t":        {"cudnnSetConvolution2dDescriptor", "cudnnSetConvolutionGroupCount", "cudnnSetConvolutionMathType", "cudnnSetConvolutionNdDescriptor", "cudnnSetConvolutionReorderType"},
		"cudnnDropoutDescriptor_t":            {"cudnnSetDropoutDescriptor"},
		"cudnnFilterDescriptor_t":             {"cudnnSetFilter4dDescriptor", "cudnnSetFilterNdDescriptor"},
		"cudnnFusedOpsConstParamPack_t":       {"cudnnSetFusedOpsConstParamPackAttribute"},
		"cudnnFusedOpsVariantParamPack_t":     {"cudnnSetFusedOpsVariantParamPackAttribute"},
		"cudnnLRNDescriptor_t":                {"cudnnSetLRNDescriptor"},
		"cudnnOpTensorDescriptor_t":           {"cudnnSetOpTensorDescriptor"},
		"cudnnPoolingDescriptor_t":            {"cudnnSetPooling2dDescriptor", "cudnnSetPoolingNdDescriptor"},
		"cudnnRNNDataDescriptor_t":            {"cudnnSetRNNDataDescriptor"},
		"cudnnRNNDescriptor_t":                {"cudnnRNNSetClip", "cudnnRNNSetClip_v8", "cudnnSetPersistentRNNPlan", "cudnnSetRNNAlgorithmDescriptor", "cudnnSetRNNBiasMode", "cudnnSetRNNDescriptor_v6", "cudnnSetRNNDescriptor_v8", "cudnnSetRNNMatrixMathType", "cudnnSetRNNPaddingMode", "cudnnSetRNNProjectionLayers"},
		"cudnnReduceTensorDescriptor_t":       {"cudnnSetReduceTensorDescriptor"},
		"cudnnSeqDataDescriptor_t":            {"cudnnSetSeqDataDescriptor"},
		"cudnnSpatialTransformerDescriptor_t": {"cudnnSetSpatialTransformerNdDescriptor"},
		"cudnnTensorDescriptor_t":             {"cudnnSetTensor", "cudnnSetTensor4dDescriptor", "cudnnSetTensor4dDescriptorEx", "cudnnSetTensorNdDescriptor", "cudnnSetTensorNdDescriptorEx"},
		"cudnnTensorTransformDescriptor_t":    {"cudnnSetTensorTransformDescriptor"},
		"unsigned":                            {"cudnnSetCallback"},
	}

	destructions = map[string][]string{
		"cudnnActivationDescriptor_t":         {"cudnnDestroyActivationDescriptor"},
		"cudnnAlgorithmPerformance_t":         {"cudnnDestroyAlgorithmPerformance"},
		"cudnnAttnDescriptor_t":               {"cudnnDestroyAttnDescriptor"},
		"cudnnBackendDescriptor_t":            {"cudnnBackendDestroyDescriptor"},
		"cudnnCTCLossDescriptor_t":            {"cudnnDestroyCTCLossDescriptor"},
		"cudnnConvolutionDescriptor_t":        {"cudnnDestroyConvolutionDescriptor"},
		"cudnnDropoutDescriptor_t":            {"cudnnDestroyDropoutDescriptor"},
		"cudnnFilterDescriptor_t":             {"cudnnDestroyFilterDescriptor"},
		"cudnnFusedOpsConstParamPack_t":       {"cudnnDestroyFusedOpsConstParamPack"},
		"cudnnFusedOpsPlan_t":                 {"cudnnDestroyFusedOpsPlan"},
		"cudnnFusedOpsVariantParamPack_t":     {"cudnnDestroyFusedOpsVariantParamPack"},
		"cudnnHandle_t":                       {"cudnnDestroy"},
		"cudnnLRNDescriptor_t":                {"cudnnDestroyLRNDescriptor"},
		"cudnnOpTensorDescriptor_t":           {"cudnnDestroyOpTensorDescriptor"},
		"cudnnPersistentRNNPlan_t":            {"cudnnDestroyPersistentRNNPlan"},
		"cudnnPoolingDescriptor_t":            {"cudnnDestroyPoolingDescriptor"},
		"cudnnRNNDataDescriptor_t":            {"cudnnDestroyRNNDataDescriptor"},
		"cudnnRNNDescriptor_t":                {"cudnnDestroyRNNDescriptor"},
		"cudnnReduceTensorDescriptor_t":       {"cudnnDestroyReduceTensorDescriptor"},
		"cudnnSeqDataDescriptor_t":            {"cudnnDestroySeqDataDescriptor"},
		"cudnnSpatialTransformerDescriptor_t": {"cudnnDestroySpatialTransformerDescriptor"},
		"cudnnTensorDescriptor_t":             {"cudnnDestroyTensorDescriptor"},
		"cudnnTensorTransformDescriptor_t":    {"cudnnDestroyTensorTransformDescriptor"},
		"cudnnAlgorithmDescriptor_t":          {"cudnnDestroyAlgorithmDescriptor"},
	}

	methods = map[string][]string{
		"cudnnHandle_t":            {"cudnnActivationBackward", "cudnnActivationForward", "cudnnAddTensor", "cudnnBatchNormalizationBackward", "cudnnBatchNormalizationForwardInference", "cudnnBatchNormalizationForwardTraining", "cudnnCTCLoss", "cudnnConvolutionBackwardBias", "cudnnConvolutionBackwardData", "cudnnConvolutionBackwardFilter", "cudnnConvolutionBiasActivationForward", "cudnnConvolutionForward", "cudnnDivisiveNormalizationBackward", "cudnnDivisiveNormalizationForward", "cudnnDropoutBackward", "cudnnDropoutForward", "cudnnDropoutGetStatesSize", "cudnnFindConvolutionBackwardDataAlgorithm", "cudnnFindConvolutionBackwardDataAlgorithmEx", "cudnnFindConvolutionBackwardFilterAlgorithm", "cudnnFindConvolutionBackwardFilterAlgorithmEx", "cudnnFindConvolutionForwardAlgorithm", "cudnnFindConvolutionForwardAlgorithmEx", "cudnnGetRNNLinLayerBiasParams", "cudnnGetRNNLinLayerMatrixParams", "cudnnGetRNNParamsSize", "cudnnGetRNNTrainingReserveSize", "cudnnGetRNNWorkspaceSize", "cudnnGetReductionIndicesSize", "cudnnGetReductionWorkspaceSize", "cudnnIm2Col", "cudnnLRNCrossChannelBackward", "cudnnLRNCrossChannelForward", "cudnnOpTensor", "cudnnPoolingBackward", "cudnnPoolingForward", "cudnnRNNBackwardData", "cudnnRNNBackwardWeights", "cudnnRNNForwardInference", "cudnnRNNForwardTraining", "cudnnReduceTensor", "cudnnScaleTensor", "cudnnSoftmaxBackward", "cudnnSoftmaxForward", "cudnnSpatialTfGridGeneratorBackward", "cudnnSpatialTfGridGeneratorForward", "cudnnSpatialTfSamplerBackward", "cudnnSpatialTfSamplerForward", "cudnnTransformTensor"},
		"cudnnTensorDescriptor_t":  {"cudnnDeriveBNTensorDescriptor", "cudnnDropoutGetReserveSpaceSize"},
		"cudnnDropoutDescriptor_t": {"cudnnRestoreDropoutDescriptor"},
	}
}
