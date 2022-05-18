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

// Apply applies a function to all the values in the tensor.
func (t *Dense) Apply(fn interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}
	if m, ok := e.(Mapper); ok {
		return m.Map(fn, t, opts...)
	}
	return nil, errors.Errorf("Execution engine %T for %v not a mapper", e, t)
}

// Reduce applies a reduction function and reduces the values along the given axis.
func (t *Dense) Reduce(fn interface{}, axis int, defaultValue interface{}) (retVal *Dense, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}

	if rd, ok := e.(Reducer); ok {
		var val Tensor
		if val, err = rd.Reduce(fn, t, axis, defaultValue); err != nil {
			err = errors.Wrapf(err, opFail, "Dense.Reduce")
			return
		}
		retVal = val.(*Dense)
		return
	}
	return nil, errors.Errorf("Engine %v is not a Reducer", e)
}
