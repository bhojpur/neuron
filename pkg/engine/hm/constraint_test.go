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

import "testing"

func TestConstraint(t *testing.T) {
	c := Constraint{
		a: TypeVariable('a'),
		b: NewFnType(TypeVariable('b'), TypeVariable('c')),
	}

	ftv := c.FreeTypeVar()
	if !ftv.Equals(TypeVarSet{TypeVariable('a'), TypeVariable('b'), TypeVariable('c')}) {
		t.Error("the free type variables of a Constraint is not as expected")
	}

	subs := mSubs{
		'a': NewFnType(proton, proton),
		'b': proton,
		'c': neutron,
	}

	c = c.Apply(subs).(Constraint)
	if !c.a.Eq(NewFnType(proton, proton)) {
		t.Errorf("c.a: %v", c)
	}

	if !c.b.Eq(NewFnType(proton, neutron)) {
		t.Errorf("c.b: %v", c)
	}
}
