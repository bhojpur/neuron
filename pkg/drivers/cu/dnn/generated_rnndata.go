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

/* Generated by gencudnn. DO NOT EDIT */

// #include <cudnn.h>
import "C"
import (
	"runtime"
	"unsafe"
)

// RNNData is a representation of cudnnRNNDataDescriptor_t.
type RNNData struct {
	internal C.cudnnRNNDataDescriptor_t

	dataType       DataType
	layout         RNNDataLayout
	maxSeqLength   int
	batchSize      int
	vectorSize     int
	seqLengthArray []int
	paddingFill    Memory
}

// NewRNNData creates a new RNNData.
func NewRNNData(dataType DataType, layout RNNDataLayout, maxSeqLength int, batchSize int, vectorSize int, seqLengthArray []int, paddingFill Memory) (retVal *RNNData, err error) {
	var internal C.cudnnRNNDataDescriptor_t
	if err := result(C.cudnnCreateRNNDataDescriptor(&internal)); err != nil {
		return nil, err
	}

	seqLengthArrayC, seqLengthArrayCManaged := ints2CIntPtr(seqLengthArray)
	defer returnManaged(seqLengthArrayCManaged)

	if err := result(C.cudnnSetRNNDataDescriptor(internal, dataType.C(), layout.C(), C.int(maxSeqLength), C.int(batchSize), C.int(vectorSize), seqLengthArrayC, unsafe.Pointer(paddingFill.Uintptr()))); err != nil {
		return nil, err
	}

	retVal = &RNNData{
		internal:       internal,
		dataType:       dataType,
		layout:         layout,
		maxSeqLength:   maxSeqLength,
		batchSize:      batchSize,
		vectorSize:     vectorSize,
		seqLengthArray: seqLengthArray,
		paddingFill:    paddingFill,
	}
	runtime.SetFinalizer(retVal, destroyRNNData)
	return retVal, nil
}

// C() returns the internal cgo representation of RNNData
func (r *RNNData) C() C.cudnnRNNDataDescriptor_t { return r.internal }

// DataType returns the internal dataType.
func (r *RNNData) DataType() DataType { return r.dataType }

// Layout returns the internal layout.
func (r *RNNData) Layout() RNNDataLayout { return r.layout }

// MaxSeqLength returns the internal maxSeqLength.
func (r *RNNData) MaxSeqLength() int { return r.maxSeqLength }

// BatchSize returns the internal batchSize.
func (r *RNNData) BatchSize() int { return r.batchSize }

// VectorSize returns the internal vectorSize.
func (r *RNNData) VectorSize() int { return r.vectorSize }

// PaddingFill returns the internal paddingFill.
func (r *RNNData) PaddingFill() Memory { return r.paddingFill }

// SeqLengthArray returns the internal `seqLengthArray` slice.
func (r *RNNData) SeqLengthArray() []int { return r.seqLengthArray }

func destroyRNNData(obj *RNNData) { C.cudnnDestroyRNNDataDescriptor(obj.internal) }
