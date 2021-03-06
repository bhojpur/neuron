//go:build linux
// +build linux

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
	"fmt"
	"io/ioutil"
	"os"
	"syscall"
	"testing"
	"testing/quick"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

type F64 float64

func newF64(f float64) *F64 { r := F64(f); return &r }

func (f *F64) Uintptr() uintptr { return uintptr(unsafe.Pointer(f)) }

func (f *F64) MemSize() uintptr { return 8 }

func (f *F64) Pointer() unsafe.Pointer { return unsafe.Pointer(f) }

func Test_FromMemory(t *testing.T) {
	fn := func(F float64) bool {
		f := newF64(F)
		T := New(WithShape(), Of(Float64), FromMemory(f.Uintptr(), f.MemSize()))
		data := T.Data().(float64)

		if data != F {
			return false
		}
		return true
	}
	if err := quick.Check(fn, &quick.Config{MaxCount: 1000000}); err != nil {
		t.Logf("%v", err)
	}

	f, err := ioutil.TempFile("", "test")
	if err != nil {
		t.Fatal(err)
	}
	// fill in with fake data
	backing := make([]byte, 8*1024*1024) // 1024*1024 matrix of float64
	asFloats := *(*[]float64)(unsafe.Pointer(&backing))
	asFloats = asFloats[: 1024*1024 : 1024*1024]
	asFloats[0] = 3.14
	asFloats[2] = 6.28
	asFloats[1024*1024-1] = 3.14
	asFloats[1024*1024-3] = 6.28
	f.Write(backing)

	// defer cleanup
	defer os.Remove(f.Name())

	// do the mmap stuff
	stat, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}

	size := int(stat.Size())
	fd := int(f.Fd())
	bs, err := syscall.Mmap(fd, 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := syscall.Munmap(bs); err != nil {
			t.Error(err)
		}
	}()
	T := New(WithShape(1024, 1024), Of(Float64), FromMemory(uintptr(unsafe.Pointer(&bs[0])), uintptr(size)))

	s := fmt.Sprintf("%v", T)
	expected := `???3.14     0  6.28     0  ...    0     0     0     0???
???   0     0     0     0  ...    0     0     0     0???
???   0     0     0     0  ...    0     0     0     0???
???   0     0     0     0  ...    0     0     0     0???
.
.
.
???   0     0     0     0  ...    0     0     0     0???
???   0     0     0     0  ...    0     0     0     0???
???   0     0     0     0  ...    0     0     0     0???
???   0     0     0     0  ...    0  6.28     0  3.14???
`
	if s != expected {
		t.Errorf("Expected mmap'd tensor to be exactly the same.")
	}

	assert.True(t, T.IsManuallyManaged())
}
