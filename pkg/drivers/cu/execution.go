package cu

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

// #include <cuda.h>
import "C"
import "unsafe"

// Function represents a CUDA function
type Function struct {
	fn C.CUfunction
}

func (fn Function) c() C.CUfunction { return fn.fn }

const pointerSize = 8 // sorry, 64 bits only.

// Launch launches a CUDA function
func (fn Function) Launch(gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) error {
	// Since Go 1.6, a cgo argument cannot have a Go pointer to Go pointer,
	// so we copy the argument values go C memory first.
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	err := result(C.cuLaunchKernel(
		fn.fn,
		C.uint(gridDimX),
		C.uint(gridDimY),
		C.uint(gridDimZ),
		C.uint(blockDimX),
		C.uint(blockDimY),
		C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		stream.c(),
		(*unsafe.Pointer)(argp),
		(*unsafe.Pointer)(nil)))
	return err
}

func offset(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + pointerSize*uintptr(i))
}

func (ctx *Ctx) LaunchKernel(fn Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {
	// Since Go 1.6, a cgo argument cannot have a Go pointer to Go pointer,
	// so we copy the argument values go C memory first.
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	f := func() error {
		return result(C.cuLaunchKernel(
			fn.fn,
			C.uint(gridDimX),
			C.uint(gridDimY),
			C.uint(gridDimZ),
			C.uint(blockDimX),
			C.uint(blockDimY),
			C.uint(blockDimZ),
			C.uint(sharedMemBytes),
			stream.c(),
			(*unsafe.Pointer)(argp),
			(*unsafe.Pointer)(nil)))
	}

	ctx.err = ctx.Do(f)
}

// STUB
func LaunchCooperativeKernel() {}

// STUB
func LauncCooperativeKernelMultiDevice() {}
