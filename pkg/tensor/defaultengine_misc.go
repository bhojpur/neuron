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
	"github.com/bhojpur/neuron/pkg/tensor/internal/storage"
	"github.com/pkg/errors"
)

func (e StdEng) Clamp(a Tensor, min, max interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, nonComplexNumberTypes); err != nil {
		return nil, errors.Wrap(err, "Clamp failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), false, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Neg")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			if err = e.E.ClampIter(typ, cloned.hdr(), ait, min, max); err != nil {
				return nil, errors.Wrapf(err, "Unable to perform Clamp")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, cloned.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.ClampIter(typ, dataReuse, rit, min, max)
			retVal = reuse
		case !safe:
			err = e.E.ClampIter(typ, dataA, ait, min, max)
			retVal = a
		default:
			cloned := a.Clone().(Tensor)
			err = e.E.ClampIter(typ, cloned.hdr(), ait, min, max)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		if err = e.E.Clamp(typ, cloned.hdr(), min, max); err != nil {
			return nil, errors.Wrapf(err, "Unable to perform Clamp")
		}
		err = e.E.Add(typ, dataReuse, cloned.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Clamp(typ, dataReuse, min, max)
		retVal = reuse
	case !safe:
		err = e.E.Clamp(typ, dataA, min, max)
		retVal = a
	default:
		cloned := a.Clone().(Tensor)
		err = e.E.Clamp(typ, cloned.hdr(), min, max)
		retVal = cloned
	}
	return
}

func (e StdEng) FMA(a, x, y Tensor) (Tensor, error) {
	return e.Mul(a, x, WithIncr(y))
}
func (e StdEng) FMAScalar(a Tensor, x interface{}, y Tensor) (Tensor, error) {
	return e.MulScalar(a, x, true, WithIncr(y))
}
