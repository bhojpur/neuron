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
import "runtime"

// LRN is a representation of cudnnLRNDescriptor_t.
type LRN struct {
	internal C.cudnnLRNDescriptor_t

	lrnN     uint
	lrnAlpha float64
	lrnBeta  float64
	lrnK     float64
}

// NewLRN creates a new LRN.
func NewLRN(lrnN uint, lrnAlpha float64, lrnBeta float64, lrnK float64) (retVal *LRN, err error) {
	var internal C.cudnnLRNDescriptor_t
	if err := result(C.cudnnCreateLRNDescriptor(&internal)); err != nil {
		return nil, err
	}

	if err := result(C.cudnnSetLRNDescriptor(internal, C.uint(lrnN), C.double(lrnAlpha), C.double(lrnBeta), C.double(lrnK))); err != nil {
		return nil, err
	}

	retVal = &LRN{
		internal: internal,
		lrnN:     lrnN,
		lrnAlpha: lrnAlpha,
		lrnBeta:  lrnBeta,
		lrnK:     lrnK,
	}
	runtime.SetFinalizer(retVal, destroyLRN)
	return retVal, nil
}

// C returns the internal cgo representation
func (l *LRN) C() C.cudnnLRNDescriptor_t { return l.internal }

// LrnN returns the internal lrnN.
func (l *LRN) LrnN() uint { return l.lrnN }

// LrnAlpha returns the internal lrnAlpha.
func (l *LRN) LrnAlpha() float64 { return l.lrnAlpha }

// LrnBeta returns the internal lrnBeta.
func (l *LRN) LrnBeta() float64 { return l.lrnBeta }

// LrnK returns the internal lrnK.
func (l *LRN) LrnK() float64 { return l.lrnK }

func destroyLRN(obj *LRN) { C.cudnnDestroyLRNDescriptor(obj.internal) }
