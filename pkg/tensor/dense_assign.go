package tensor

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

import (
	"github.com/pkg/errors"
)

func overlaps(a, b DenseTensor) bool {
	if a.cap() == 0 || b.cap() == 0 {
		return false
	}
	aarr := a.arr()
	barr := b.arr()
	if aarr.Uintptr() == barr.Uintptr() {
		return true
	}
	aptr := aarr.Uintptr()
	bptr := barr.Uintptr()

	capA := aptr + uintptr(cap(aarr.Header.Raw))
	capB := bptr + uintptr(cap(barr.Header.Raw))

	switch {
	case aptr < bptr:
		if bptr < capA {
			return true
		}
	case aptr > bptr:
		if aptr < capB {
			return true
		}
	}
	return false
}

func assignArray(dest, src DenseTensor) (err error) {
	// var copiedSrc bool

	if src.IsScalar() {
		panic("HELP")
	}

	dd := dest.Dims()
	sd := src.Dims()

	dstrides := dest.Strides()
	sstrides := src.Strides()

	var ds, ss int
	ds = dstrides[0]
	if src.IsVector() {
		ss = sstrides[0]
	} else {
		ss = sstrides[sd-1]
	}

	// when dd == 1, and the strides point in the same direction
	// we copy to a temporary if there is an overlap of data
	if ((dd == 1 && sd >= 1 && ds*ss < 0) || dd > 1) && overlaps(dest, src) {
		// create temp
		// copiedSrc = true
	}

	// broadcast src to dest for raw iteration
	tmpShape := Shape(BorrowInts(sd))
	tmpStrides := BorrowInts(len(src.Strides()))
	copy(tmpShape, src.Shape())
	copy(tmpStrides, src.Strides())
	defer ReturnInts(tmpShape)
	defer ReturnInts(tmpStrides)

	if sd > dd {
		tmpDim := sd
		for tmpDim > dd && tmpShape[0] == 1 {
			tmpDim--

			// this is better than tmpShape = tmpShape[1:]
			// because we are going to return these ints later
			copy(tmpShape, tmpShape[1:])
			copy(tmpStrides, tmpStrides[1:])
		}
	}

	var newStrides []int
	if newStrides, err = BroadcastStrides(dest.Shape(), tmpShape, dstrides, tmpStrides); err != nil {
		err = errors.Wrapf(err, "BroadcastStrides failed")
		return
	}
	dap := dest.Info()
	sap := MakeAP(tmpShape, newStrides, src.Info().o, src.Info().Δ)

	diter := newFlatIterator(dap)
	siter := newFlatIterator(&sap)
	_, err = copyDenseIter(dest, src, diter, siter)
	sap.zeroOnly() // cleanup, but not entirely because tmpShape and tmpStrides are separately cleaned up.  Don't double free
	return
}
