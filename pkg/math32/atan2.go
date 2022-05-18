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

// Atan2 returns the arc tangent of y/x, using the signs of the two to determine
// the quadrant of the return value.
// Special cases are (in order):
// 	Atan2(y, NaN) = NaN
// 	Atan2(NaN, x) = NaN
// 	Atan2(+0, x>=0) = +0
// 	Atan2(-0, x>=0) = -0
// 	Atan2(+0, x<=-0) = +Pi
// 	Atan2(-0, x<=-0) = -Pi
// 	Atan2(y>0, 0) = +Pi/2
// 	Atan2(y<0, 0) = -Pi/2
// 	Atan2(+Inf, +Inf) = +Pi/4
// 	Atan2(-Inf, +Inf) = -Pi/4
// 	Atan2(+Inf, -Inf) = 3Pi/4
// 	Atan2(-Inf, -Inf) = -3Pi/4
// 	Atan2(y, +Inf) = 0
// 	Atan2(y>0, -Inf) = +Pi
// 	Atan2(y<0, -Inf) = -Pi
// 	Atan2(+Inf, x) = +Pi/2
// 	Atan2(-Inf, x) = -Pi/2
func Atan2(y, x float32) float32 {
	// special cases
	switch {
	case IsNaN(y) || IsNaN(x):
		return NaN()
	case y == 0:
		if x >= 0 && !Signbit(x) {
			return Copysign(0, y)
		}
		return Copysign(Pi, y)
	case x == 0:
		return Copysign(Pi/2, y)
	case IsInf(x, 0):
		if IsInf(x, 1) {
			switch {
			case IsInf(y, 0):
				return Copysign(Pi/4, y)
			default:
				return Copysign(0, y)
			}
		}
		switch {
		case IsInf(y, 0):
			return Copysign(3*Pi/4, y)
		default:
			return Copysign(Pi, y)
		}
	case IsInf(y, 0):
		return Copysign(Pi/2, y)
	}

	// Call atan and determine the quadrant.
	q := Atan(y / x)
	if x < 0 {
		if q <= 0 {
			return q + Pi
		}
		return q - Pi
	}
	return q
}
