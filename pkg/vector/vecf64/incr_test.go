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

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func makeIncr(size int) []float64 {
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = 100
	}
	return retVal
}

func TestIncrAdd(t *testing.T) {
	assert := assert.New(t)

	a := Range(0, niceprime)
	incr := makeIncr(len(a))

	correct := Range(0, niceprime)
	for i := range correct {
		correct[i] = correct[i] + correct[i] + incr[i]
	}

	IncrAdd(a, a, incr)
	assert.Equal(correct, incr)

	b := Range(niceprime, 2*niceprime)
	for i := range correct {
		correct[i] = a[i] + b[i] + incr[i]
	}

	IncrAdd(a, b, incr)
	assert.Equal(correct, incr)
}

func TestIncrSub(t *testing.T) {
	assert := assert.New(t)

	a := Range(0, niceprime)
	incr := makeIncr(len(a))

	correct := make([]float64, niceprime)
	copy(correct, incr)
	IncrSub(a, a, incr)
	assert.Equal(correct, incr)

	b := Range(niceprime, 2*niceprime)
	for i := range correct {
		correct[i] = a[i] - b[i] + incr[i]
	}

	IncrSub(a, b, incr)
	assert.Equal(correct, incr)
}

func TestIncrMul(t *testing.T) {
	assert := assert.New(t)

	a := Range(0, niceprime)
	incr := makeIncr(len(a))

	correct := Range(0, niceprime)
	for i := range correct {
		correct[i] = correct[i]*correct[i] + incr[i]
	}
	IncrMul(a, a, incr)
	assert.Equal(correct, incr)

	b := Range(niceprime, 2*niceprime)
	for i := range correct {
		correct[i] = a[i]*b[i] + incr[i]
	}

	IncrMul(a, b, incr)
	assert.Equal(correct, incr)
}

func TestIncrDiv(t *testing.T) {
	assert := assert.New(t)

	a := []float64{1, 2, 4, 8, 10}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	copy(correct, a)
	for i := range correct {
		correct[i] = correct[i]/correct[i] + incr[i]
	}
	IncrDiv(a, a, incr)
	assert.Equal(correct, incr)

	b := []float64{2, 4, 8, 16, 20}
	incr = makeIncr(len(a))
	for i := range correct {
		correct[i] = a[i]/b[i] + incr[i]
	}

	IncrDiv(a, b, incr)
	assert.Equal(correct, incr)

	// division by 0
	b = make([]float64, len(a))
	IncrDiv(a, b, incr)
	for _, v := range incr {
		if !math.IsInf(v, 0) && !math.IsNaN(v) {
			t.Error("Expected Inf or NaN")
		}
	}
}

func TestIncrPow(t *testing.T) {
	a := []float64{0, 1, 2, 3, 4}
	b := []float64{0, 1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, 5)
	for i := range correct {
		correct[i] = math.Pow(a[i], b[i]) + incr[i]
	}
	IncrPow(a, b, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrScale(t *testing.T) {
	a := []float64{0, 1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, 5)
	for i := range correct {
		correct[i] = a[i]*5 + incr[i]
	}
	IncrScale(a, 5, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrScaleInv(t *testing.T) {
	a := []float64{0, 1, 2, 4, 6}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = a[i]/2 + incr[i]
	}
	IncrScaleInv(a, 2, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrScaleInvR(t *testing.T) {
	a := []float64{0, 1, 2, 4, 6}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = 2/a[i] + incr[i]
	}
	IncrScaleInvR(a, 2, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrTrans(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = a[i] + float64(1) + incr[i]
	}
	IncrTrans(a, 1, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrTransInv(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = a[i] - float64(1) + incr[i]
	}
	IncrTransInv(a, 1, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrTransInvR(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = float64(1) - a[i] + incr[i]
	}
	IncrTransInvR(a, 1, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrPowOf(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = math.Pow(a[i], 5) + incr[i]
	}
	IncrPowOf(a, 5, incr)
	assert.Equal(t, correct, incr)
}

func TestIncrPowOfR(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	incr := makeIncr(len(a))

	correct := make([]float64, len(a))
	for i := range correct {
		correct[i] = math.Pow(5, a[i]) + incr[i]
	}
	IncrPowOfR(a, 5, incr)
	assert.Equal(t, correct, incr)
}
