package cublas

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

// #include <cublas_v2.h>
import "C"
import (
	"sync"

	"github.com/bhojpur/neuron/pkg/drivers/cu"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
)

var (
	_ blas.Float32    = &Standard{}
	_ blas.Float64    = &Standard{}
	_ blas.Complex64  = &Standard{}
	_ blas.Complex128 = &Standard{}
)

// BLAS is the interface for all cuBLAS implementaions
type BLAS interface {
	cu.Context
	blas.Float32
	blas.Float64
	blas.Complex64
	blas.Complex128
}

// Standard is the standard cuBLAS handler.
// By default it assumes that the data is in  RowMajor, DESPITE the fact that cuBLAS
// takes ColMajor only. This is done for the ease of use of developers writing in Go.
//
// Use New to create a new BLAS handler.
// Use the various ConsOpts to set the options
type Standard struct {
	h C.cublasHandle_t
	o Order
	m PointerMode
	e error

	cu.Context
	dataOnDev bool

	sync.Mutex
}

func New(opts ...ConsOpt) *Standard {
	var handle C.cublasHandle_t
	if err := status(C.cublasCreate(&handle)); err != nil {
		panic(err)
	}

	impl := &Standard{
		h: handle,
	}

	for _, opt := range opts {
		opt(impl)
	}

	return impl
}

func (impl *Standard) Init(opts ...ConsOpt) error {
	impl.Lock()
	defer impl.Unlock()

	var handle C.cublasHandle_t
	if err := status(C.cublasCreate(&handle)); err != nil {
		return errors.Wrapf(err, "Failed to initialize Standard implementation of CUBLAS")
	}
	impl.h = handle

	for _, opt := range opts {
		opt(impl)
	}
	return nil
}

func (impl *Standard) Err() error { return impl.e }

func (impl *Standard) Close() error {
	impl.Lock()
	defer impl.Unlock()

	var empty C.cublasHandle_t
	if impl.h == empty {
		return nil
	}
	if err := status(C.cublasDestroy(impl.h)); err != nil {
		return err
	}
	impl.h = empty
	impl.Context = nil
	return nil
}
