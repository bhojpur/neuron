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

func TestBitMap(t *testing.T) {
	assert := assert.New(t)
	bm := NewBitMap(64)
	assert.Equal(1, len(bm.n))

	track := uint64(0)
	for i := 0; i < 64; i++ {
		bm.Set(i)
		track |= uint64(1) << uint64(i)
		assert.Equal(track, bm.n[0])
		assert.Equal(true, bm.IsSet(i))
		if i < 63 {
			assert.Equal(false, bm.IsSet(i+1))
		} else {
			fails := func() {
				bm.IsSet(i + 1)
			}
			assert.Panics(fails)
		}
	}

	for i := 0; i < 64; i++ {
		bm.Clear(i)
		track &= ^(uint64(1) << uint64(i))
		assert.Equal(track, bm.n[0])
		assert.Equal(false, bm.IsSet(i))
	}

	bm = NewBitMap(124)
	assert.Equal(2, len(bm.n))

	track0 := uint64(0)
	track1 := uint64(0)
	for i := 0; i < 128; i++ {
		if i < 124 {
			bm.Set(i)
		} else {
			fails := func() {
				bm.Set(i)
			}
			assert.Panics(fails)
		}
		if i < 64 {
			track0 |= uint64(1) << uint64(i)
			assert.Equal(track0, bm.n[0])
			assert.Equal(true, bm.IsSet(i))
		} else if i > 123 {
			fails := func() {
				bm.IsSet(i)
			}
			assert.Panics(fails)
		} else {
			track1 |= uint64(1) << uint64(i-64)
			assert.Equal(track1, bm.n[1])
			assert.Equal(true, bm.IsSet(i))
		}

		if i < 123 {
			assert.Equal(false, bm.IsSet(i+1))
		} else {
			fails := func() {
				bm.IsSet(i + 1)
			}
			assert.Panics(fails)
		}
	}

	for i := 48; i < 70; i++ {
		bm.Clear(i)
	}

	for i := 48; i < 70; i++ {
		assert.Equal(false, bm.IsSet(i))
	}

	fails := func() {
		bm.Clear(125)
	}
	assert.Panics(fails)

	// idiots section!
	bm = NewBitMap(3)
	fails = func() {
		bm.Set(-1)
	}
	assert.Panics(fails)

	fails = func() {
		bm.Set(3)
	}
	assert.Panics(fails)

}
