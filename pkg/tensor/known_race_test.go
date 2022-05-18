//go:build ignore && !race
// +build ignore,!race

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
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

// This test will fail the `go test -race`.
//
// This is because FromMemory() will use uintptr in a way that is incorrect according to the checkptr directive of Go 1.14+
//
// Though it's incorrect, it's the only way to use heterogenous, readable memory (i.e. CUDA).
func TestFromMemory(t *testing.T) {
	// dummy memory - this could be an externally malloc'd memory, or a mmap'ed file.
	// but here we're just gonna let Go manage memory.
	s := make([]float64, 100)
	ptr := uintptr(unsafe.Pointer(&s[0]))
	size := uintptr(100 * 8)

	T := New(Of(Float32), WithShape(50, 4), FromMemory(ptr, size))
	if len(T.Float32s()) != 200 {
		t.Error("expected 200 Float32s")
	}
	assert.Equal(t, make([]float32, 200), T.Data())
	assert.True(t, T.IsManuallyManaged(), "Unamanged %v |%v | q: %v", ManuallyManaged, T.flag, (T.flag>>ManuallyManaged)&MemoryFlag(1))

	fail := func() { New(FromMemory(ptr, size), Of(Float32)) }
	assert.Panics(t, fail, "Expected bad New() call to panic")
}
