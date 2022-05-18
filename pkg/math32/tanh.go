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

// The original C code, the long comment, and the constants
// below were from http://netlib.sandia.gov/cephes/cmath/tanh.c,
// available from http://www.netlib.org/cephes/single.tgz.
// The go code is a simplified version of the original C.
// tanhf.c
//
// Hyperbolic tangent
//
//
//
// SYNOPSIS:
//
// float x, y, tanhf();
//
// y = tanhf( x );
//
//
//
// DESCRIPTION:
//
// Returns hyperbolic tangent of argument in the range MINLOG to
// MAXLOG.
//
// A polynomial approximation is used for |x| < 0.625.
// Otherwise,
//
// tanh(x) = sinh(x)/cosh(x) = 1  -  2/(exp(2x) + 1).
//
//
//
// ACCURACY:
//
// Relative error:
// arithmetic   domain     # trials      peak         rms
// IEEE      -2,2        100000      1.3e-7      2.6e-8
//
//

/*
Cephes Math Library Release 2.2:  June, 1992
Copyright 1984, 1987, 1989, 1992 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

/* Single precision hyperbolic tangent
 * test interval: [-0.625, +0.625]
 * trials: 10000
 * peak relative error: 7.2e-8
 * rms relative error: 2.6e-8
 */

func Tanh(x float32) float32 {
	const MAXLOG = 88.02969187150841
	z := Abs(x)
	switch {
	case z > 0.5*MAXLOG:
		if x < 0 {
			return -1
		}
		return 1
	case z >= 0.625:
		s := Exp(z + z)
		z = 1 - 2/(s+1)
		if x < 0 {
			z = -z
		}
	default:
		if x == 0 {
			return x
		}
		s := x * x
		z = ((((-5.70498872745e-3*s+2.06390887954e-2)*s-5.37397155531e-2)*s+1.33314422036e-1)*s-3.33332819422e-1)*s*x + x
	}
	return z
}
