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
	"github.com/bhojpur/neuron/pkg/tensor/internal/execution"
	"github.com/pkg/errors"
)

// StdEng is the default execution engine that comes with the tensors. To use other execution engines, use the WithEngine construction option.
type StdEng struct {
	execution.E
}

// makeArray allocates a slice for the array
func (e StdEng) makeArray(arr *array, t Dtype, size int) {
	arr.Raw = malloc(t, size)
	arr.t = t
}

func (e StdEng) AllocAccessible() bool             { return true }
func (e StdEng) Alloc(size int64) (Memory, error)  { return nil, noopError{} }
func (e StdEng) Free(mem Memory, size int64) error { return nil }
func (e StdEng) Memset(mem Memory, val interface{}) error {
	if ms, ok := mem.(MemSetter); ok {
		return ms.Memset(val)
	}
	return errors.Errorf("Cannot memset %v with StdEng", mem)
}

func (e StdEng) Memclr(mem Memory) {
	if z, ok := mem.(Zeroer); ok {
		z.Zero()
	}
	return
}

func (e StdEng) Memcpy(dst, src Memory) error {
	switch dt := dst.(type) {
	case *array:
		switch st := src.(type) {
		case *array:
			copyArray(dt, st)
			return nil
		case arrayer:
			copyArray(dt, st.arrPtr())
			return nil
		}
	case arrayer:
		switch st := src.(type) {
		case *array:
			copyArray(dt.arrPtr(), st)
			return nil
		case arrayer:
			copyArray(dt.arrPtr(), st.arrPtr())
			return nil
		}
	}
	return errors.Errorf("Failed to copy %T %T", dst, src)
}

func (e StdEng) Accessible(mem Memory) (Memory, error) { return mem, nil }

func (e StdEng) WorksWith(order DataOrder) bool { return true }

func (e StdEng) checkAccessible(t Tensor) error {
	if !t.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, t)
	}
	return nil
}
