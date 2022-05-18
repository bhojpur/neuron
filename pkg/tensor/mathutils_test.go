package tensor

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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDivmod(t *testing.T) {
	as := []int{0, 1, 2, 3, 4, 5}
	bs := []int{1, 2, 3, 3, 2, 3}
	qs := []int{0, 0, 0, 1, 2, 1}
	rs := []int{0, 1, 2, 0, 0, 2}

	for i, a := range as {
		b := bs[i]
		eq := qs[i]
		er := rs[i]

		q, r := divmod(a, b)
		if q != eq {
			t.Errorf("Expected %d / %d to equal %d. Got %d instead", a, b, eq, q)
		}
		if r != er {
			t.Errorf("Expected %d %% %d to equal %d. Got %d instead", a, b, er, r)
		}
	}

	assert := assert.New(t)
	fail := func() {
		divmod(1, 0)
	}
	assert.Panics(fail)
}
