package storage

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
)

// Header is runtime representation of a slice. It's a cleaner version of reflect.SliceHeader.
// With this, we wouldn't need to keep the uintptr.
// This usually means additional pressure for the GC though, especially when passing around Headers
type Header struct {
	Raw []byte
}

// TypedLen returns the length of data as if it was a slice of type t
func (h *Header) TypedLen(t reflect.Type) int {
	return len(h.Raw) / int(t.Size())
}

func Copy(t reflect.Type, dst, src *Header) int {
	copied := copy(dst.Raw, src.Raw)
	return copied / int(t.Size())
}

func CopySliced(t reflect.Type, dst *Header, dstart, dend int, src *Header, sstart, send int) int {
	dstBA := dst.Raw
	srcBA := src.Raw
	size := int(t.Size())

	ds := dstart * size
	de := dend * size
	ss := sstart * size
	se := send * size
	copied := copy(dstBA[ds:de], srcBA[ss:se])
	return copied / size
}

func SwapCopy(a, b *Header) {
	for i := range a.Raw {
		a.Raw[i], b.Raw[i] = b.Raw[i], a.Raw[i]
	}
}

func Fill(t reflect.Type, dst, src *Header) int {
	dstBA := dst.Raw
	srcBA := src.Raw
	size := int(t.Size())
	lenSrc := len(srcBA)

	dstart := 0
	for {
		copied := copy(dstBA[dstart:], srcBA)
		dstart += copied
		if copied < lenSrc {
			break
		}
	}
	return dstart / size
}

func CopyIter(t reflect.Type, dst, src *Header, diter, siter Iterator) int {
	dstBA := dst.Raw
	srcBA := src.Raw
	size := int(t.Size())

	var idx, jdx, i, j, count int
	var err error
	for {
		if idx, err = diter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				panic(err)
			}
			break
		}
		if jdx, err = siter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				panic(err)
			}
			break
		}
		i = idx * size
		j = jdx * size
		copy(dstBA[i:i+size], srcBA[j:j+size])
		// dstBA[i : i+size] = srcBA[j : j+size]
		count++
	}
	return count
}

// Element gets the pointer of ith element
func ElementAt(i int, base unsafe.Pointer, typeSize uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(base) + uintptr(i)*typeSize)
}

// AsByteSlice takes a slice of anything and returns a casted-as-byte-slice view of it.
// This function panics if input is not a slice.
func AsByteSlice(x interface{}) []byte {
	xV := reflect.ValueOf(x)
	xT := reflect.TypeOf(x).Elem() // expects a []T

	hdr := reflect.SliceHeader{
		Data: xV.Pointer(),
		Len:  xV.Len() * int(xT.Size()),
		Cap:  xV.Cap() * int(xT.Size()),
	}
	return *(*[]byte)(unsafe.Pointer(&hdr))
}

func FromMemory(ptr uintptr, memsize uintptr) []byte {
	hdr := reflect.SliceHeader{
		Data: ptr,
		Len:  int(memsize),
		Cap:  int(memsize),
	}
	return *(*[]byte)(unsafe.Pointer(&hdr))
}
