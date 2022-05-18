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
import (
	"runtime"
)

type TensorDescriptor struct {
	internal C.cudnnTensorDescriptor_t // ptr to struct

	// internal data for fast
	format   TensorFormat
	dataType DataType
	shape    []int // NCHW format for 4-tensors
	strides  []int
}

func NewTensorDescriptor(format TensorFormat, dt DataType, shape, strides []int) (*TensorDescriptor, error) {
	var internal C.cudnnTensorDescriptor_t
	if err := result(C.cudnnCreateTensorDescriptor(&internal)); err != nil {
		return nil, err
	}

	retVal := &TensorDescriptor{
		internal: internal,
		format:   format,
		dataType: dt,
		shape:    shape,
		strides:  strides,
	}

	runtime.SetFinalizer(retVal, destroyTensor)
	if err := retVal.set(internal); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *TensorDescriptor) set(internal C.cudnnTensorDescriptor_t) error {

	switch len(t.shape) {
	case 4:
		n, c, h, w := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
		if len(t.strides) == 4 {
			// use explicit
			NStrides, CStrides, HStrides, WStrides := t.strides[0], t.strides[1], t.strides[2], t.strides[3]
			res := C.cudnnSetTensor4dDescriptorEx(internal, t.dataType.C(),
				C.int(n), C.int(c), C.int(h), C.int(w),
				C.int(NStrides), C.int(CStrides), C.int(HStrides), C.int(WStrides),
			)
			return result(res)
		}

		// otherwise the strides will be calculated by cudnn
		res := C.cudnnSetTensor4dDescriptor(internal, t.format.C(), t.dataType.C(),
			C.int(n), C.int(c), C.int(h), C.int(w),
		)
		return result(res)
	default:
		if len(t.strides) > 0 {
			dimA, dimAManaged := ints2CIntPtr(t.shape)
			defer returnManaged(dimAManaged)
			strideA, strideAManaged := ints2CIntPtr(t.strides)
			defer returnManaged(strideAManaged)
			// NO, there is no confusion here. Ex is used to set tensor without strides. Silly nVidia.
			res := C.cudnnSetTensorNdDescriptor(internal, t.dataType.C(),
				C.int(len(t.shape)), dimA, strideA)
			return result(res)
		}
		dimA, dimAManaged := ints2CIntPtr(t.shape)
		defer returnManaged(dimAManaged)
		res := C.cudnnSetTensorNdDescriptorEx(internal, t.format.C(), t.dataType.C(),
			C.int(len(t.shape)), dimA)
		return result(res)
	}
}

func (t *TensorDescriptor) Format() TensorFormat { return t.format }
func (t *TensorDescriptor) DataType() DataType   { return t.dataType }
func (t *TensorDescriptor) Shape() []int         { return cloneShape(t.shape) }
func (t *TensorDescriptor) Strides() []int       { return cloneShape(t.strides) }

func destroyTensor(obj *TensorDescriptor) { C.cudnnDestroyTensorDescriptor(obj.internal) }
