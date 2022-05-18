//go:build !avx && !sse
// +build !avx,!sse

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

import "github.com/bhojpur/neuron/pkg/math32"

// Add performs a̅ + b̅. a̅ will be clobbered
func Add(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v + b[i]
	}
}

// Sub performs a̅ - b̅. a̅ will be clobbered
func Sub(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v - b[i]
	}
}

// Mul performs a̅ × b̅. a̅ will be clobbered
func Mul(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v * b[i]
	}
}

// Div performs a̅ ÷ b̅. a̅ will be clobbered
func Div(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		if b[i] == 0 {
			a[i] = math32.Inf(0)
			continue
		}

		a[i] = v / b[i]
	}
}

// Sqrt performs √a̅ elementwise. a̅ will be clobbered
func Sqrt(a []float32) {
	for i, v := range a {
		a[i] = math32.Sqrt(v)
	}
}

// InvSqrt performs 1/√a̅ elementwise. a̅ will be clobbered
func InvSqrt(a []float32) {
	for i, v := range a {
		a[i] = float32(1) / math32.Sqrt(v)
	}
}

// Sum sums a slice of float32 and returns a float32
func Sum(a []float32) float32 {
	var retVal float32
	for _, v := range a {
		retVal += v
	}
	return retVal
}
