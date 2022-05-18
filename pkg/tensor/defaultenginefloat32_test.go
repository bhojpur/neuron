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
	"testing/quick"
)

func TestFloat32Engine_makeArray(t *testing.T) {

	// the uint16 is just to make sure that tests are correctly run.
	// we don't want the quicktest to randomly generate a size that is so large
	// that Go takes a long time just to allocate. We'll test the other sizes (like negative numbers)
	// after the quick test.
	f := func(sz uint16) bool {
		size := int(sz)
		e := Float32Engine{StdEng{}}
		dt := Float32
		arr := array{}

		e.makeArray(&arr, dt, size)

		if len(arr.Raw) != size*4 {
			t.Errorf("Expected raw to be size*4. Got %v instead", len(arr.Raw))
			return false
		}
		v, ok := arr.Data().([]float32)
		if !ok {
			t.Errorf("Expected v to be []float32. Got %T instead", arr.Data())
			return false
		}

		if len(v) != size {
			return false
		}
		return true
	}

	if err := quick.Check(f, nil); err != nil {
		t.Errorf("Quick test failed %v", err)
	}

}
