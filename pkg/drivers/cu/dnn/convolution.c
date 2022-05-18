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

#include <cudnn.h>

cudnnStatus_t gocudnnNewConvolution(cudnnConvolutionDescriptor_t *retVal,
	cudnnMathType_t mathType, const int groupCount,
	const int size, const int* padding,
	const int* filterStrides,
	const int* dilation,
	cudnnConvolutionMode_t convolutionMode, cudnnDataType_t dataType) {

	cudnnStatus_t status ;
	status = cudnnCreateConvolutionDescriptor(retVal);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	status = cudnnSetConvolutionMathType(*retVal, mathType);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	status = cudnnSetConvolutionGroupCount(*retVal, groupCount);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	int padH;
	int padW;
	int u;
	int v;
	int dilationH;
	int dilationW;
	switch (size) {
	case 0:
	case 1:
		return CUDNN_STATUS_BAD_PARAM;
	case 2:
		padH = padding[0];
		padW = padding[1];
		u = filterStrides[0];
		v = filterStrides[1];
		dilationH = dilation[0];
		dilationW = dilation[1];

		status = cudnnSetConvolution2dDescriptor(*retVal,
			padH, padW,
			u, v,
			dilationH, dilationW,
			convolutionMode, dataType);
		break;
	default:
		status = cudnnSetConvolutionNdDescriptor(*retVal, size, padding, filterStrides, dilation, convolutionMode, dataType);
		break;
	}
	return status;
}