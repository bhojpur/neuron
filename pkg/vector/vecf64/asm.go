//go:build sse || avx
// +build sse avx

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

// Add performs a̅ + b̅. a̅ will be clobbered
func Add(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	addAsm(a, b)
}
func addAsm(a, b []float64)

// Sub performs a̅ - b̅. a̅ will be clobbered
func Sub(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	subAsm(a, b)
}
func subAsm(a, b []float64)

// Mul performs a̅ × b̅. a̅ will be clobbered
func Mul(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	mulAsm(a, b)
}
func mulAsm(a, b []float64)

// Div performs a̅ ÷ b̅. a̅ will be clobbered
func Div(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	divAsm(a, b)
}
func divAsm(a, b []float64)

// Sqrt performs √a̅ elementwise. a̅ will be clobbered
func Sqrt(a []float64)

// InvSqrt performs 1/√a̅ elementwise. a̅ will be clobbered
func InvSqrt(a []float64)

/*
func Pow(a, b []float64)
*/

/*
func Scale(s float64, a []float64)
func ScaleFrom(s float64, a []float64)
func Trans(s float64, a []float64)
func TransFrom(s float64, a []float64)
func Power(s float64, a []float64)
func PowerFrom(s float64, a []float64)
*/
