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

func Exp(x float32) float32 {
	if haveArchExp {
		return archExp(x)
	}
	return exp(x)
}

func exp(x float32) float32 {
	const (
		Ln2Hi = float32(6.9313812256e-01)
		Ln2Lo = float32(9.0580006145e-06)
		Log2e = float32(1.4426950216e+00)

		Overflow  = 7.09782712893383973096e+02
		Underflow = -7.45133219101941108420e+02
		NearZero  = 1.0 / (1 << 28) // 2**-28

		LogMax = 0x42b2d4fc // The bitmask of log(FLT_MAX), rounded down.  This value is the largest input that can be passed to exp() without producing overflow.
		LogMin = 0x42aeac50 // The bitmask of |log(REAL_FLT_MIN)|, rounding down

	)
	// hx := Float32bits(x) & uint32(0x7fffffff)

	// special cases
	switch {
	case IsNaN(x) || IsInf(x, 1):
		return x
	case IsInf(x, -1):
		return 0
	case x > Overflow:
		return Inf(1)
	case x < Underflow:
		// case hx > LogMax:
		// 	return Inf(1)
		// case x < 0 && hx > LogMin:
		return 0
	case -NearZero < x && x < NearZero:
		return 1 + x
	}

	// reduce; computed as r = hi - lo for extra precision.
	var k int
	switch {
	case x < 0:
		k = int(Log2e*x - 0.5)
	case x > 0:
		k = int(Log2e*x + 0.5)
	}
	hi := x - float32(k)*Ln2Hi
	lo := float32(k) * Ln2Lo

	// compute
	return expmulti(hi, lo, k)
}

// Exp2 returns 2**x, the base-2 exponential of x.
//
// Special cases are the same as Exp.
func Exp2(x float32) float32 {
	if haveArchExp2 {
		return archExp2(x)
	}
	return exp2(x)
}

func exp2(x float32) float32 {
	const (
		Ln2Hi = 6.9313812256e-01
		Ln2Lo = 9.0580006145e-06

		Overflow  = 1.0239999999999999e+03
		Underflow = -1.0740e+03
	)

	// special cases
	switch {
	case IsNaN(x) || IsInf(x, 1):
		return x
	case IsInf(x, -1):
		return 0
	case x > Overflow:
		return Inf(1)
	case x < Underflow:
		return 0
	}

	// argument reduction; x = r×lg(e) + k with |r| ≤ ln(2)/2.
	// computed as r = hi - lo for extra precision.
	var k int
	switch {
	case x > 0:
		k = int(x + 0.5)
	case x < 0:
		k = int(x - 0.5)
	}
	t := x - float32(k)
	hi := t * Ln2Hi
	lo := -t * Ln2Lo

	// compute
	return expmulti(hi, lo, k)
}

// exp1 returns e**r × 2**k where r = hi - lo and |r| ≤ ln(2)/2.
func expmulti(hi, lo float32, k int) float32 {
	const (
		P1 = float32(1.6666667163e-01)  /* 0x3e2aaaab */
		P2 = float32(-2.7777778450e-03) /* 0xbb360b61 */
		P3 = float32(6.6137559770e-05)  /* 0x388ab355 */
		P4 = float32(-1.6533901999e-06) /* 0xb5ddea0e */
		P5 = float32(4.1381369442e-08)  /* 0x3331bb4c */
	)

	r := hi - lo
	t := r * r
	c := r - t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
	y := 1 - ((lo - (r*c)/(2-c)) - hi)
	// TODO(rsc): make sure Ldexp can handle boundary k
	return Ldexp(y, k)
}
