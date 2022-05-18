package main

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
	"reflect"
	"unsafe"

	"github.com/bhojpur/neuron/pkg/drivers/cu"
	cublas "github.com/bhojpur/neuron/pkg/drivers/cu/blas"
	"github.com/bhojpur/neuron/pkg/tensor"
)

type Engine struct {
	tensor.StdEng
	ctx cu.Context
	*cublas.Standard
}

func newEngine() *Engine {
	ctx := cu.NewContext(cu.Device(0), cu.SchedAuto)
	blas := cublas.New(cublas.WithContext(ctx))
	return &Engine{
		ctx:      ctx,
		Standard: blas,
	}
}

func (e *Engine) AllocAccessible() bool { return true }

func (e *Engine) Alloc(size int64) (tensor.Memory, error) {
	return e.ctx.MemAllocManaged(size, cu.AttachGlobal)
}

func (e *Engine) AllocFlags() (tensor.MemoryFlag, tensor.DataOrder) {
	return tensor.MakeMemoryFlag(tensor.ManuallyManaged), tensor.ColMajor
}

func (e *Engine) Free(mem tensor.Memory, size int64) error {
	e.ctx.MemFree(mem.(cu.DevicePtr))
	return nil
}

func (e *Engine) Memset(mem tensor.Memory, val interface{}) error {
	panic("not implemented")
}

func (e *Engine) Memclr(mem tensor.Memory) {
	panic("not implemented")
}

func (e *Engine) Memcpy(dst tensor.Memory, src tensor.Memory) error {
	panic("not implemented")
}

func (e *Engine) Accessible(mem tensor.Memory) (tensor.Memory, error) {
	// panic("not implemented")
	size := mem.MemSize()
	retVal := make([]byte, int(size))
	e.ctx.MemcpyDtoH(unsafe.Pointer(&retVal[0]), cu.DevicePtr(mem.Uintptr()), int64(size))
	l := int(size / 8)
	foo2 := &reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&retVal[0])),
		Len:  l,
		Cap:  l,
	}
	return *(*foomem)(unsafe.Pointer(foo2)), e.ctx.Error()

}

func (e *Engine) WorksWith(order tensor.DataOrder) bool { return true }

func (e *Engine) NonStdAlloc() {}

func (e *Engine) ContextErr() error { return e.ctx.Error() }

type foomem []float64

func (m foomem) Uintptr() uintptr { return uintptr(unsafe.Pointer(&m[0])) }
func (m foomem) MemSize() uintptr { return uintptr(len(m) * 8) }
