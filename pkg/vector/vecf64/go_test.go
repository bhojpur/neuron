//go:build !sse && !avx
// +build !sse,!avx

package vecf64

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

/*
IMPORTANT NOTE:

Currently Div does not handle division by zero correctly. It returns a NaN instead of +Inf
*/

import (
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestDiv(t *testing.T) {
	assert := assert.New(t)

	a := Range(0, niceprime-1)

	correct := Range(0, niceprime-1)
	for i := range correct {
		correct[i] = 1
	}
	Div(a, a)
	assert.Equal(correct[1:], a[1:])
	assert.Equal(true, math.IsInf(a[0], 0), "a[0] is: %v", a[0])

	b := Range(niceprime, 2*niceprime-1)
	for i := range correct {
		correct[i] = a[i] / b[i]
	}

	Div(a, b)
	assert.Equal(correct[1:], a[1:])
	assert.Equal(true, math.IsInf(a[0], 0), "a[0] is: %v", a[0])

	/* Weird Corner Cases*/

	for i := 1; i < 65; i++ {
		a = Range(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			b = Range(i, 2*i)
			correct = make([]float64, i)
			for j := range correct {
				correct[j] = a[j] / b[j]
			}
			Div(a, b)
			assert.Equal(correct[1:], a[1:])
		}
	}

}

func TestSqrt(t *testing.T) {
	assert := assert.New(t)

	a := Range(0, niceprime-1)

	correct := Range(0, niceprime-1)
	for i, v := range correct {
		correct[i] = math.Sqrt(v)
	}
	Sqrt(a)
	assert.Equal(correct, a)

	// negatives
	a = []float64{-1, -2, -3, -4}
	Sqrt(a)

	for _, v := range a {
		if !math.IsNaN(v) {
			t.Error("Expected NaN")
		}
	}

	/* Weird Corner Cases*/
	for i := 1; i < 65; i++ {
		a = Range(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			correct = make([]float64, i)
			for j := range correct {
				correct[j] = math.Sqrt(a[j])
			}
			Sqrt(a)
			assert.Equal(correct, a)
		}
	}
}

func TestInvSqrt(t *testing.T) {

	assert := assert.New(t)
	a := Range(0, niceprime-1)

	correct := Range(0, niceprime-1)
	for i, v := range correct {
		correct[i] = 1.0 / math.Sqrt(v)
	}
	InvSqrt(a)
	assert.Equal(correct[1:], a[1:])
	if !math.IsInf(a[0], 0) {
		t.Error("1/0 should be +Inf or -Inf")
	}

	// Weird Corner Cases

	for i := 1; i < 65; i++ {
		a = Range(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			correct = make([]float64, i)
			for j := range correct {
				correct[j] = 1.0 / math.Sqrt(a[j])
			}
			InvSqrt(a)
			assert.Equal(correct[1:], a[1:], "i = %d, %v", i, Range(0, i))
			if !math.IsInf(a[0], 0) {
				t.Error("1/0 should be +Inf or -Inf")
			}
		}
	}
}
