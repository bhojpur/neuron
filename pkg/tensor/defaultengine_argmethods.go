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

import "github.com/pkg/errors"

func (e StdEng) Argmax(t Tensor, axis int) (retVal Tensor, err error) {

	switch tt := t.(type) {
	case DenseTensor:
		return e.argmaxDenseTensor(tt, axis)
	default:
		return nil, errors.Errorf(typeNYI, "StdEng.Argmax", t)
	}
}

func (e StdEng) argmaxDenseTensor(t DenseTensor, axis int) (retVal *Dense, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, opFail, "Argmax")
	}

	if axis >= len(t.Shape()) {
		return nil, errors.Errorf(dimMismatch, len(t.Shape()), axis)
	}

	dataA := t.hdr()
	typ := t.rtype()

	// SPECIAL CASE: FLAT ARGMAX
	if axis == AllAxes {
		var index int
		if mt, ok := t.(MaskedTensor); ok && mt.IsMasked() {
			if index = e.E.ArgmaxFlatMasked(typ, dataA, mt.Mask()); index == -1 {
				return nil, errors.Errorf("t is not supported - %T of %v", t, t.Dtype())
			}
		} else {
			if index = e.E.ArgmaxFlat(typ, dataA); index == -1 {
				return nil, errors.Errorf("t is not supported -  %T of %v", t, t.Dtype())
			}
		}
		return New(FromScalar(index)), nil
	}

	// ARGMAX ALONG AXIS

	var indices []int
	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}

	// be a good citizen - borrow and return, since we're only using this AP to figure out the moves
	newAP, _, err := t.Info().T(axes...)
	if _, ok := err.(NoOpError); !ok && err != nil {
		return
	} else if ok {
		t.Info().CloneTo(&newAP)
	}

	it := IteratorFromDense(t)
	iteratorLoadAP(it, &newAP)

	lastSize := it.Shape()[len(it.Shape())-1]
	newShape := it.Shape().Clone()
	newShape = newShape[:len(newShape)-1]

	// cleanup
	defer func() {
		newAP.zero()
		ReturnInts(newShape)
	}()

	if mt, ok := t.(MaskedTensor); ok && mt.IsMasked() {
		mask := mt.Mask()
		if indices, err = e.E.ArgmaxIterMasked(typ, dataA, mask, it, lastSize); err != nil {
			return
		}
	} else {
		if indices, err = e.E.ArgmaxIter(typ, dataA, it, lastSize); err != nil {
			return
		}
	}

	return New(WithShape(newShape...), WithBacking(indices)), nil
}

func (e StdEng) Argmin(t Tensor, axis int) (retVal Tensor, err error) {

	switch tt := t.(type) {
	case DenseTensor:
		return e.argminDenseTensor(tt, axis)
	default:
		return nil, errors.Errorf(typeNYI, "StdEng.Argmin", t)
	}
}

func (e StdEng) argminDenseTensor(t DenseTensor, axis int) (retVal *Dense, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, opFail, "Argmin")
	}

	if axis >= len(t.Shape()) {
		return nil, errors.Errorf(dimMismatch, len(t.Shape()), axis)
	}

	dataA := t.hdr()
	typ := t.rtype()

	// SPECIAL CASE: FLAT ARGMAX
	if axis == AllAxes {
		var index int
		if mt, ok := t.(MaskedTensor); ok && mt.IsMasked() {
			if index = e.E.ArgminFlatMasked(typ, dataA, mt.Mask()); index == -1 {
				return nil, errors.Errorf("t is not supported - %T of %v", t, t.Dtype())
			}
		} else {
			if index = e.E.ArgminFlat(typ, dataA); index == -1 {
				return nil, errors.Errorf("t is not supported -  %T of %v", t, t.Dtype())
			}
		}
		return New(FromScalar(index)), nil
	}

	// ARGMAX ALONG AXIS

	var indices []int
	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}

	// be a good citizen - borrow and return, since we're only using this AP to figure out the moves
	newAP, _, err := t.Info().T(axes...)
	if _, ok := err.(NoOpError); !ok && err != nil {
		return
	} else if ok {
		newAP = t.Info().Clone()
	}

	it := IteratorFromDense(t)
	iteratorLoadAP(it, &newAP)

	lastSize := it.Shape()[len(it.Shape())-1]
	newShape := it.Shape().Clone()
	newShape = newShape[:len(newShape)-1]

	// cleanup
	defer func() {
		newAP.zero()
		ReturnInts(newShape)
	}()

	if mt, ok := t.(MaskedTensor); ok && mt.IsMasked() {
		mask := mt.Mask()
		if indices, err = e.E.ArgminIterMasked(typ, dataA, mask, it, lastSize); err != nil {
			return
		}
	} else {
		if indices, err = e.E.ArgminIter(typ, dataA, it, lastSize); err != nil {
			return
		}
	}

	return New(WithShape(newShape...), WithBacking(indices)), nil
}
