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
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestCopy(t *testing.T) {
	// A longer than B
	a := headerFromSlice([]int{0, 1, 2, 3, 4})
	b := headerFromSlice([]int{10, 11})
	copied := Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 2, copied)
	assert.Equal(t, []int{10, 11, 2, 3, 4}, a.Ints())

	// B longer than A
	a = headerFromSlice([]int{10, 11})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 2, copied)
	assert.Equal(t, []int{0, 1}, a.Ints())

	// A is empty
	a = headerFromSlice([]int{})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 0, copied)

	// B is empty
	a = headerFromSlice([]int{0, 1, 2, 3, 4})
	b = headerFromSlice([]int{})
	copied = Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 0, copied)
	assert.Equal(t, []int{0, 1, 2, 3, 4}, a.Ints())
}

func TestFill(t *testing.T) {
	// A longer than B
	a := headerFromSlice([]int{0, 1, 2, 3, 4})
	b := headerFromSlice([]int{10, 11})
	copied := Fill(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 5, copied)
	assert.Equal(t, []int{10, 11, 10, 11, 10}, a.Ints())

	// B longer than A
	a = headerFromSlice([]int{10, 11})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Fill(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 2, copied)
	assert.Equal(t, []int{0, 1}, a.Ints())
}

func headerFromSlice(x interface{}) Header {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	xV := reflect.ValueOf(x)
	size := uintptr(xV.Len()) * xT.Elem().Size()
	return Header{
		Raw: FromMemory(xV.Pointer(), size),
	}
}
