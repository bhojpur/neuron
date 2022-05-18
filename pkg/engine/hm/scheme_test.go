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

func TestSchemeBasics(t *testing.T) {
	s := new(Scheme)
	s.tvs = TypeVarSet{'a', 'b'}
	s.t = NewFnType(TypeVariable('c'), proton)

	sub := mSubs{
		'a': proton,
		'b': neutron,
		'c': electron,
	}

	s2 := s.Apply(nil).(*Scheme)
	if s2 != s {
		t.Errorf("Different pointers")
	}

	s2 = s.Apply(sub).(*Scheme)
	if s2 != s {
		t.Errorf("Different pointers")
	}

	if !s.tvs.Equals(TypeVarSet{'a', 'b'}) {
		t.Error("TypeVarSet mutated")
	}

	if !s.t.Eq(NewFnType(electron, proton)) {
		t.Error("Application failed")
	}

	s = new(Scheme)
	s.tvs = TypeVarSet{'a', 'b'}
	s.t = NewFnType(TypeVariable('c'), proton)

	ftv := s.FreeTypeVar()

	if !ftv.Equals(TypeVarSet{'c'}) {
		t.Errorf("Expected ftv: {'c'}. Got %v instead", ftv)
	}

	// format
	if fmt.Sprintf("%v", s) != "∀[a, b]: c → proton" {
		t.Errorf("Scheme format is wrong.: Got %q", fmt.Sprintf("%v", s))
	}

	// Polytype scheme.Type
	T, isMono := s.Type()
	if isMono {
		t.Errorf("%v is supposed to be a polytype. It shouldn't return true", s)
	}
	if !T.Eq(NewFnType(TypeVariable('c'), proton)) {
		t.Error("Wrong type returned by scheme")
	}
}

func TestSchemeNormalize(t *testing.T) {
	s := new(Scheme)
	s.tvs = TypeVarSet{'c', 'z', 'd'}
	s.t = NewFnType(TypeVariable('a'), TypeVariable('c'))

	err := s.Normalize()
	if err != nil {
		t.Error(err)
	}

	if !s.tvs.Equals(TypeVarSet{'a', 'b'}) {
		t.Errorf("Expected: TypeVarSet{'a','b'}. Got: %v", s.tvs)
	}
}
