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

/*
It provides basic constants and mathematical functions for float32 types.
At its core, it's mostly just a wrapper in form of float32(math.XXX). This
applies to the following functions:
	Acos
	Acosh
	Asin
	Asinh
	Atan
	Atan2
	Atanh
	Cbrt
	Cos
	Cosh
	Erfc
	Gamma
	J0
	J1
	Jn
	Log10
	Log1p
	Log2
	Logb
	Pow10
	Sin
	Sinh
	Tan
	Y0
	Y1

Everything else is a float32 implementation. Implementation schedule is sporadic
an uncertain. But eventually all functions will be replaced
*/
