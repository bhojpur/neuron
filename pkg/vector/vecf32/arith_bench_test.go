//go:build !sse && !avx
// +build !sse,!avx

package vecf32

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

	"github.com/bhojpur/neuron/pkg/math32"
)

/* BENCHMARKS */

func _vanillaVecAdd(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

func BenchmarkVecAdd(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		Add(x, y)
	}
}

func BenchmarkVanillaVecAdd(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecAdd(x, y)
	}
}

func _vanillaVecSub(a, b []float32) {
	for i := range a {
		a[i] -= b[i]
	}
}

func BenchmarkVecSub(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		Sub(x, y)
	}
}

func BenchmarkVanillaVecSub(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecSub(x, y)
	}
}

func _vanillaVecMul(a, b []float32) {
	for i := range a {
		a[i] *= b[i]
	}
}

func BenchmarkVecMul(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		Mul(x, y)
	}
}

func BenchmarkVanillaVecMul(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecMul(x, y)
	}
}

func _vanillaVecDiv(a, b []float32) {
	for i := range a {
		a[i] /= b[i]
	}
}

func BenchmarkVecDiv(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		Div(x, y)
	}
}

func BenchmarkVanillaVecDiv(b *testing.B) {
	x := Range(0, niceprime)
	y := Range(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecDiv(x, y)
	}
}

func _vanillaVecSqrt(a []float32) {
	for i, v := range a {
		a[i] = math32.Sqrt(v)
	}
}

func BenchmarkVecSqrt(b *testing.B) {
	x := Range(0, niceprime)

	for n := 0; n < b.N; n++ {
		Sqrt(x)
	}
}

func BenchmarkVanillaVecSqrt(b *testing.B) {
	x := Range(0, niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecSqrt(x)
	}
}

func _vanillaVecInverseSqrt(a []float32) {
	for i, v := range a {
		a[i] = 1.0 / math32.Sqrt(v)
	}
}

func BenchmarkVecInvSqrt(b *testing.B) {
	x := Range(0, niceprime)

	for n := 0; n < b.N; n++ {
		InvSqrt(x)
	}
}

func BenchmarkVanillaVecInvSqrt(b *testing.B) {
	x := Range(0, niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecInverseSqrt(x)
	}
}
