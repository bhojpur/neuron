package cudnn

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

// #include <stdint.h>
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

var cintsize, gointsize int

func init() {
	cintsize = int(C.sizeof_int)
	gointsize = int(unsafe.Sizeof(int(1)))
}

// int32sPool is a pool of Go accessible []int32.
// The internal pool is necessary for setting a bunch of values (usually shapes)
var int32sPool = &sync.Pool{
	New: func() interface{} { return make([]int32, 0, 8) },
}

// returnManaged returns any managed slices to the pool.
func returnManaged(a interface{}) {
	if a == nil {
		return
	}

	switch x := a.(type) {
	case []int32:
		for i := range x {
			x[i] = 0
		}
		x = x[:0]
		int32sPool.Put(x)
	}
}

// ints2CIntPtr takes a []int and returns a C pointer to the slice.
// On architectures where the Go int and the C int sizes are different,
// a slice of C-int-size-equivalent ints will be allocated. This is called "managed".
// The C pointer will be to that newly allocated slice. The `managed` slice will also be returned.
func ints2CIntPtr(a []int) (cPtr *C.int, managed interface{}) {
	if cintsize == gointsize {
		return (*C.int)(unsafe.Pointer(&a[0])), nil
	}
	switch {
	case cintsize == 4 && gointsize == 8:
		b := int32sPool.Get().([]int32)
		for _, v := range a {
			b = append(b, int32(v))
		}
		return (*C.int)(unsafe.Pointer(&b[0])), b
	default:
		panic(fmt.Sprintf("UNHANDLED: cintsize: %v gointsize: %v", cintsize, gointsize))
	}
}

func int32s2CInt32Ptr(a []int32) (cPtr *C.int32_t) {
	return (*C.int32_t)(unsafe.Pointer(&a[0]))
}

func uint32s2CUint32Ptr(a []uint32) (cPtr *C.uint32_t) {
	return (*C.uint32_t)(unsafe.Pointer(&a[0]))
}
