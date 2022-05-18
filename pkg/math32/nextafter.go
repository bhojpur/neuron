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

// Nextafter returns the next representable float32 value after x towards y.
//
// Special cases are:
//	Nextafter32(x, x)   = x
//	Nextafter32(NaN, y) = NaN
//	Nextafter32(x, NaN) = NaN
func Nextafter(x, y float32) (r float32) {
	switch {
	case IsNaN(x) || IsNaN(y): // special case
		r = float32(NaN())
	case x == y:
		r = x
	case x == 0:
		r = float32(Copysign(Float32frombits(1), y))
	case (y > x) == (x > 0):
		r = Float32frombits(Float32bits(x) + 1)
	default:
		r = Float32frombits(Float32bits(x) - 1)
	}
	return
}
