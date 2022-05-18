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
	"reflect"
	"testing"
)

type Float16 uint16

func TestRegisterType(t *testing.T) {
	dt := Dtype{reflect.TypeOf(Float16(0))}
	RegisterFloat(dt)

	if err := typeclassCheck(dt, floatTypes); err != nil {
		t.Errorf("Expected %v to be in floatTypes: %v", dt, err)
	}
	if err := typeclassCheck(dt, numberTypes); err != nil {
		t.Errorf("Expected %v to be in numberTypes: %v", dt, err)
	}
	if err := typeclassCheck(dt, ordTypes); err != nil {
		t.Errorf("Expected %v to be in ordTypes: %v", dt, err)
	}
	if err := typeclassCheck(dt, eqTypes); err != nil {
		t.Errorf("Expected %v to be in eqTypes: %v", dt, err)
	}

}

func TestDtypeConversions(t *testing.T) {
	for k, v := range reverseNumpyDtypes {
		if npdt, err := v.numpyDtype(); npdt != k {
			t.Errorf("Expected %v to return numpy dtype of %q. Got %q instead", v, k, npdt)
		} else if err != nil {
			t.Errorf("Error: %v", err)
		}
	}
	dt := Dtype{reflect.TypeOf(Float16(0))}
	if _, err := dt.numpyDtype(); err == nil {
		t.Errorf("Expected an error when passing in type unknown to np")
	}

	for k, v := range numpyDtypes {
		if dt, err := fromNumpyDtype(v); dt != k {
			// special cases
			if Int.Size() == 4 && v == "i4" && dt == Int {
				continue
			}
			if Int.Size() == 8 && v == "i8" && dt == Int {
				continue
			}

			if Uint.Size() == 4 && v == "u4" && dt == Uint {
				continue
			}
			if Uint.Size() == 8 && v == "u8" && dt == Uint {
				continue
			}
			t.Errorf("Expected %q to return %v. Got %v instead", v, k, dt)
		} else if err != nil {
			t.Errorf("Error: %v", err)
		}
	}
	if _, err := fromNumpyDtype("EDIUH"); err == nil {
		t.Error("Expected error when nonsense is passed into fromNumpyDtype")
	}
}
