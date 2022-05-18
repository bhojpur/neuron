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
	"unsafe"

	"github.com/google/uuid"
)

// Device is the representation of a CUDA device
type Device int

const (
	CPU       Device = -1
	BadDevice Device = -2
)

// Name returns the name of the device.
//
// Wrapper over cuDeviceGetName: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f
func (d Device) Name() (string, error) {
	size := 256
	buf := make([]byte, 256)
	cstr := C.CString(string(buf))
	defer C.free(unsafe.Pointer(cstr))
	if err := result(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(d))); err != nil {
		return "", err
	}
	return C.GoString(cstr), nil
}

// UUID returns the UUID of the device
//
// Wrapper over cuDeviceGetUuid: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4
func (d Device) UUID() (retVal uuid.UUID, err error) {
	ptr := &retVal
	if err = result(C.cuDeviceGetUuid((*C.CUuuid)(unsafe.Pointer(ptr)), C.CUdevice(d))); err != nil {
		return retVal, err
	}
	return retVal, nil
}

// String implementes fmt.Stringer (and runtime.stringer)
func (d Device) String() string {
	if d == CPU {
		return "CPU"
	}
	if d < 0 {
		return "Invalid Device"
	}
	return fmt.Sprintf("GPU(%d)", int(d))
}

// IsGPU returns true if the device is a GPU.
func (d Device) IsGPU() bool {
	if d < 0 {
		return false
	}
	return true
}
