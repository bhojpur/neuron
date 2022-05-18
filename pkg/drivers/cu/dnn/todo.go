package cudnn

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

// #include <cudnn.h>
import "C"

// TODO

/*
func (ctx *Context) GetRNNLinLayerBiasParams(rnnDesc *RNN, pseudoLayer int, xDesc *TensorDescriptor, wDesc *Filter, w Memory, linLayerID int, linLayerBiasDesc *Filter, linLayerBias TODO) error {
	// call cudnnGetRNNLinLayerBiasParams
	return result(C.cudnnGetRNNLinLayerBiasParams(ctx.internal, rnnDesc.internal, C.int(pseudoLayer), xDesc.internal, wDesc.internal, unsafe.Pointer(w.Uintptr()), C.int(linLayerID), linLayerBiasDesc.internal, linLayerBias))
}
func (ctx *Context) GetRNNLinLayerMatrixParams(rnnDesc *RNN, pseudoLayer int, xDesc *TensorDescriptor, wDesc *Filter, w Memory, linLayerID int, linLayerMatDesc *Filter, linLayerMat TODO) error {
	// call cudnnGetRNNLinLayerMatrixParams
	return result(C.cudnnGetRNNLinLayerMatrixParams(ctx.internal, rnnDesc.internal, C.int(pseudoLayer), xDesc.internal, wDesc.internal, unsafe.Pointer(w.Uintptr()), C.int(linLayerID), linLayerMatDesc.internal, linLayerMat))
}

// Input. Handle to a previously created cuDNN context. For more information, see cudnnHandle_t.
func (co *Context) CTCLoss(probsDesc *TensorDescriptor, probs Memory, hostLabels TODO, hostLabelLengths TODO, hostInputLengths TODO, costs Memory, gradientsDesc *TensorDescriptor, gradients Memory, algo CTCLossAlgo, ctcLossDesc *CTCLoss, workspace Memory, workSpaceSizeInBytes uintptr) error {
	// DOUBLECHECK: "cudnnCTCLoss" returns Memory type in Parameter 8
	// call cudnnCTCLoss
	return result(C.cudnnCTCLoss(co.internal, probsDesc.internal, unsafe.Pointer(probs.Uintptr()), hostLabels, hostLabelLengths, hostInputLengths, unsafe.Pointer(costs.Uintptr()), gradientsDesc.internal, unsafe.Pointer(gradients.Uintptr()), algo.C(), ctcLossDesc.internal, unsafe.Pointer(workspace.Uintptr()), C.size_t(workSpaceSizeInBytes)))
}
*/
