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

/* THIS IS TEMPORARY, UNTIL I FIGURE OUT A BETTER WAY TO SET UNION VALUES IN GO */

cudnnAlgorithm_t makeConvFwdAlgo(cudnnConvolutionFwdAlgo_t algo) {
	cudnnAlgorithm_t retVal;
	retVal.algo.convFwdAlgo = algo;
	return retVal;
}

cudnnAlgorithm_t makeConvBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t algo){
	cudnnAlgorithm_t retVal;
	retVal.algo.convBwdFilterAlgo = algo;
	return retVal;
}


cudnnAlgorithm_t makeConvBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t algo){
	cudnnAlgorithm_t retVal;
	retVal.algo.convBwdDataAlgo = algo;
	return retVal;
}

cudnnAlgorithm_t makeRNNAlgo(cudnnRNNAlgo_t algo) {
       	cudnnAlgorithm_t retVal;
	retVal.algo.RNNAlgo = algo;
	return retVal;
}

cudnnAlgorithm_t makeCTCLossAlgo(cudnnCTCLossAlgo_t algo) {
	cudnnAlgorithm_t retVal;
	retVal.algo.CTCLossAlgo = algo;
	return retVal;
}