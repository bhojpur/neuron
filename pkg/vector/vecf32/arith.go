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

// Pow performs  elementwise
//		a̅ ^ b̅
func Pow(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		switch b[i] {
		case 0:
			a[i] = float32(1)
		case 1:
			a[i] = v
		case 2:
			a[i] = v * v
		case 3:
			a[i] = v * v * v
		default:
			a[i] = math32.Pow(v, b[i])
		}
	}
}

func Mod(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = math32.Mod(v, b[i])
	}
}

// Scale multiplies all values in the slice by the scalar. It performs elementwise
// 		a̅ * s
func Scale(a []float32, s float32) {
	for i, v := range a {
		a[i] = v * s
	}
}

// ScaleInv divides all values in the slice by the scalar. It performs elementwise
// 		a̅ / s
func ScaleInv(a []float32, s float32) {
	Scale(a, 1/s)
}

/// ScaleInvR divides all numbers in the slice by a scalar
// 		s / a̅
func ScaleInvR(a []float32, s float32) {
	for i, v := range a {
		a[i] = s / v
	}
}

// Trans adds all the values in the slice by a scalar
// 		a̅ + s
func Trans(a []float32, s float32) {
	for i, v := range a {
		a[i] = v + s
	}
}

// TransInv subtracts all the values in the slice by a scalar
//		a̅ - s
func TransInv(a []float32, s float32) {
	Trans(a, -s)
}

// TransInvR subtracts all the numbers in a slice from a scalar
//	 s - a̅
func TransInvR(a []float32, s float32) {
	for i, v := range a {
		a[i] = s - v
	}
}

// PowOf performs elementwise
//		a̅ ^ s
func PowOf(a []float32, s float32) {
	for i, v := range a {
		a[i] = math32.Pow(v, s)
	}
}

// PowOfR performs elementwise
//		s ^ a̅
func PowOfR(a []float32, s float32) {
	for i, v := range a {
		a[i] = math32.Pow(s, v)
	}
}

// Max takes two slices, a̅ + b̅, and compares them elementwise. The highest value is put into a̅.
func Max(a, b []float32) {
	b = b[:len(a)]

	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

// Min takes two slices, a̅ + b̅ and compares them elementwise. The lowest value is put into a̅.
func Min(a, b []float32) {
	b = b[:len(a)]

	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

/* REDUCTION RELATED */

// MaxOf finds the max of a []float32. it panics if the slice is empty
func MaxOf(a []float32) (retVal float32) {
	if len(a) < 1 {
		panic("Cannot find the max of an empty slice")
	}
	return Reduce(max, a[0], a[1:]...)
}

// MinOf finds the max of a []float32. it panics if the slice is empty
func MinOf(a []float32) (retVal float32) {
	if len(a) < 1 {
		panic("Cannot find the min of an empty slice")
	}
	return Reduce(min, a[0], a[1:]...)
}

// Argmax returns the index of the min in a slice
func Argmax(a []float32) int {
	var f float32
	var max int
	var set bool
	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		// TODO: Maybe error instead of this?
		if math32.IsNaN(v) || math32.IsInf(v, 1) {
			max = i
			f = v
			break
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

// Argmin returns the index of the min in a slice
func Argmin(a []float32) int {
	var f float32
	var min int
	var set bool
	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}

		// TODO: Maybe error instead of this?
		if math32.IsNaN(v) || math32.IsInf(v, -1) {
			min = i
			f = v
			break
		}

		if v < f {
			min = i
			f = v
		}
	}
	return min
}

/* FUNCTION VARIABLES */

var (
	add = func(a, b float32) float32 { return a + b }
	// sub = func(a, b float32) float32 { return a - b }
	// mul = func(a, b float32) float32 { return a * b }
	// div = func(a, b float32) float32 { return a / b }
	// mod = func(a, b float32) float32 { return math32.Mod(a, b) }

	min = func(a, b float32) float32 {
		if a < b {
			return a
		}
		return b
	}

	max = func(a, b float32) float32 {
		if a > b {
			return a
		}
		return b
	}
)
