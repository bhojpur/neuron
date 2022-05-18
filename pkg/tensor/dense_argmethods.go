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

/* Argmax */

// Argmax finds the index of the max value along the axis provided
func (t *Dense) Argmax(axis int) (retVal *Dense, err error) {
	e := t.e
	switch am := e.(type) {
	case denseArgmaxer:
		return am.argmaxDenseTensor(t, axis)
	case Argmaxer:
		var ret Tensor
		var ok bool
		if ret, err = am.Argmax(t, axis); err != nil {
			return nil, errors.Wrapf(err, opFail, "Argmax")
		}
		if retVal, ok = ret.(*Dense); !ok {
			return nil, errors.Errorf(extractionFail, "*Dense", ret)
		}
		return
	}
	return nil, errors.New("Engine does not suport Argmax")
}

/* Argmin */

// Argmin finds the index of the min value along the axis provided
func (t *Dense) Argmin(axis int) (retVal *Dense, err error) {
	e := t.e
	switch am := e.(type) {
	case denseArgminer:
		return am.argminDenseTensor(t, axis)
	case Argminer:
		var ret Tensor
		var ok bool
		if ret, err = am.Argmin(t, axis); err != nil {
			return nil, errors.Wrapf(err, opFail, "Argmax")
		}
		if retVal, ok = ret.(*Dense); !ok {
			return nil, errors.Errorf(extractionFail, "*Dense", ret)
		}
		return
	}
	return nil, errors.New("Engine does not suport Argmax")
}
