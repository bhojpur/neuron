package math32

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

// Round returns the nearest integer, rounding half away from zero.
//
// Special cases are:
//	Round(±0) = ±0
//	Round(±Inf) = ±Inf
//	Round(NaN) = NaN
func Round(x float32) float32 {
	// Round is a faster implementation of:
	//
	// func Round(x float64) float64 {
	//   t := Trunc(x)
	//   if Abs(x-t) >= 0.5 {
	//     return t + Copysign(1, x)
	//   }
	//   return t
	// }
	bits := Float32bits(x)
	e := uint(bits>>shift) & mask
	if e < bias {
		// Round abs(x) < 1 including denormals.
		bits &= signMask // +-0
		if e == bias-1 {
			bits |= uvone // +-1
		}
	} else if e < bias+shift {
		// Round any abs(x) >= 1 containing a fractional component [0,1).
		//
		// Numbers with larger exponents are returned unchanged since they
		// must be either an integer, infinity, or NaN.
		const half = 1 << (shift - 1)
		e -= bias
		bits += half >> e
		bits &^= fracMask >> e
	}
	return Float32frombits(bits)
}

// RoundToEven returns the nearest integer, rounding ties to even.
//
// Special cases are:
//	RoundToEven(±0) = ±0
//	RoundToEven(±Inf) = ±Inf
//	RoundToEven(NaN) = NaN
func RoundToEven(x float32) float32 {
	// RoundToEven is a faster implementation of:
	//
	// func RoundToEven(x float64) float64 {
	//   t := math.Trunc(x)
	//   odd := math.Remainder(t, 2) != 0
	//   if d := math.Abs(x - t); d > 0.5 || (d == 0.5 && odd) {
	//     return t + math.Copysign(1, x)
	//   }
	//   return t
	// }
	bits := Float32bits(x)
	e := uint(bits>>shift) & mask
	if e >= bias {
		// Round abs(x) >= 1.
		// - Large numbers without fractional components, infinity, and NaN are unchanged.
		// - Add 0.499.. or 0.5 before truncating depending on whether the truncated
		//   number is even or odd (respectively).
		const halfMinusULP = (1 << (shift - 1)) - 1
		e -= bias
		bits += (halfMinusULP + (bits>>(shift-e))&1) >> e
		bits &^= fracMask >> e
	} else if e == bias-1 && bits&fracMask != 0 {
		// Round 0.5 < abs(x) < 1.
		bits = bits&signMask | uvone // +-1
	} else {
		// Round abs(x) <= 0.5 including denormals.
		bits &= signMask // +-0
	}
	return Float32frombits(bits)
}
