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

//#include <cuda.h>
import "C"
import (
	"fmt"

	"github.com/pkg/errors"
)

// DevicePtr is a pointer to the device memory. It is equivalent to CUDA's CUdeviceptr
type DevicePtr uintptr

func (d DevicePtr) String() string { return fmt.Sprintf("0x%x", uintptr(d)) }

func (d DevicePtr) AddressRange() (size int64, base DevicePtr, err error) {
	var s C.size_t
	var b C.CUdeviceptr
	if err = result(C.cuMemGetAddressRange(&b, &s, C.CUdeviceptr(d))); err != nil {
		err = errors.Wrapf(err, "MemGetAddressRange")
		return
	}
	return int64(s), DevicePtr(b), nil
}

// Uintptr returns the pointer in form of a uintptr
func (d DevicePtr) Uintptr() uintptr { return uintptr(d) }

// IsCUDAMemory returns true.
func (d DevicePtr) IsCUDAMemory() bool { return true }
