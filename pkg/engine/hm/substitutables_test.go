package hm

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
	"fmt"
	"testing"
)

func TestConstraints(t *testing.T) {
	cs := Constraints{
		{TypeVariable('a'), proton},
		{TypeVariable('b'), proton},
	}
	correct := TypeVarSet{'a', 'b'}

	ftv := cs.FreeTypeVar()
	for _, v := range correct {
		if !ftv.Contains(v) {
			t.Errorf("Expected free type vars to contain %v", v)
			break
		}
	}

	sub := mSubs{
		'a': neutron,
	}

	cs = cs.Apply(sub).(Constraints)
	if cs[0].a != neutron {
		t.Error("Expected neutron")
	}
	if cs[0].b != proton {
		t.Error("Expected proton")
	}

	if cs[1].a != TypeVariable('b') {
		t.Error("There was nothing to substitute b with")
	}
	if cs[1].b != proton {
		t.Error("Expected proton")
	}

	if fmt.Sprintf("%v", cs) != "Constraints[{neutron = proton}, {b = proton}]" {
		t.Errorf("Error in formatting cs")
	}

}

func TestTypes_Contains(t *testing.T) {
	ts := Types{TypeVariable('a'), proton}

	if !ts.Contains(TypeVariable('a')) {
		t.Error("Expected ts to contain 'a'")
	}

	if !ts.Contains(proton) {
		t.Error("Expected ts to contain proton")
	}

	if ts.Contains(neutron) {
		t.Error("ts shouldn't contain neutron")
	}
}
