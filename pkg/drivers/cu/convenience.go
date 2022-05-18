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
import (
	"log"
	"unsafe"

	"github.com/pkg/errors"
)

// This file lists all the convenience functions and methods, not necessarily stuff that is covered in the API

// MemoryType returns the MemoryType of the memory
func (mem DevicePtr) MemoryType() (typ MemoryType, err error) {
	var p unsafe.Pointer
	if p, err = mem.PtrAttribute(MemoryTypeAttr); err != nil {
		return
	}
	t := *(*uint64)(p)
	typ = MemoryType(byte(t))
	return
}

// MemSize returns the size of the memory slab in bytes. Returns 0 if errors occured
func (mem DevicePtr) MemSize() uintptr {
	size, _, err := mem.AddressRange()
	if err != nil {
		log.Printf("MEMSIZE ERR %v", err)
	}
	return uintptr(size)
}

// ComputeCapability returns the compute capability of the device.
// This method is a convenience method for the deprecated API call cuDeviceComputeCapability.
func (d Device) ComputeCapability() (major, minor int, err error) {
	var attrs []int
	if attrs, err = d.Attributes(ComputeCapabilityMajor, ComputeCapabilityMinor); err != nil {
		err = errors.Wrapf(err, "Failed to get ComputeCapability")
		return
	}
	major = attrs[0]
	minor = attrs[1]
	return
}
