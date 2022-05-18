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
// #include "batch.h"
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

/* COMMON PATTERNS */

// Attributes gets multiple attributes as provided
func (dev Device) Attributes(attrs ...DeviceAttribute) ([]int, error) {
	if len(attrs) == 0 {
		return nil, nil
	}
	cAttrs := make([]C.CUdevice_attribute, len(attrs))
	cRetVal := make([]C.int, len(attrs))
	size := C.int(len(attrs))

	for i, v := range attrs {
		cAttrs[i] = C.CUdevice_attribute(v)
	}

	err := result(C.cuDeviceGetAttributes(&cRetVal[0], &cAttrs[0], size, C.CUdevice(dev)))
	retVal := make([]int, len(attrs))
	for i, v := range cRetVal {
		retVal[i] = int(v)
	}

	return retVal, err
}

// LaunchAndSync launches the kernel and synchronizes the context
func (fn Function) LaunchAndSync(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) error {
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	err := result(C.cuLaunchAndSync(
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

// AllocAndCopy abstracts away the common pattern of allocating and then copying a Go slice to the GPU
func AllocAndCopy(p unsafe.Pointer, bytesize int64) (DevicePtr, error) {
	if bytesize == 0 {
		return 0, errors.Wrapf(InvalidValue, "Cannot allocate memory with size 0")
	}

	var d C.CUdeviceptr
	if err := result(C.cuAllocAndCopy(&d, p, C.size_t(bytesize))); err != nil {
		return 0, errors.Wrapf(err, "AllocAndCopy")
	}
	return DevicePtr(d), nil
}
