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

var subsTests = []struct {
	op string
	tv TypeVariable
	t  Type

	ok   bool
	size int
}{
	{"get", TypeVariable('a'), nil, false, 0},
	{"add", TypeVariable('a'), proton, true, 1},
	{"get", TypeVariable('a'), proton, true, 1},
	{"add", TypeVariable('a'), neutron, true, 1},
	{"get", TypeVariable('a'), neutron, true, 1},
	{"rem", TypeVariable('b'), nil, false, 1},
	{"rem", TypeVariable('a'), nil, false, 0},
	{"add", TypeVariable('a'), proton, true, 1},
	{"add", TypeVariable('b'), proton, true, 2},
	{"add", TypeVariable('c'), proton, true, 3},
}

func testSubs(t *testing.T, sub Subs) {
	var T Type
	var ok bool
	for _, sts := range subsTests {
		switch sts.op {
		case "get":
			if T, ok = sub.Get(sts.tv); ok != sts.ok {
				t.Errorf("Expected Get to return %t. Got a value of %v instead", sts.ok, T)
			}
		case "add":
			sub = sub.Add(sts.tv, sts.t)
		case "rem":
			sub = sub.Remove(sts.tv)
		}

		if sub.Size() != sts.size {
			t.Errorf("Inconsistent size. Want %d. Got %d", sts.size, sub.Size())
		}
	}

	// Iter
	correct := []Substitution{
		{TypeVariable('a'), proton},
		{TypeVariable('b'), proton},
		{TypeVariable('c'), proton},
	}

	for _, s := range sub.Iter() {
		var found bool
		for _, c := range correct {
			if s.T == c.T && s.Tv == c.Tv {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Testing of %T: cannot find %v in Range", sub, s)
		}
	}

	// Clone
	cloned := sub.Clone()
	cloned = cloned.Add(TypeVariable('a'), photon)
	gt, ok := sub.Get(TypeVariable('a'))
	if !ok {
		t.Errorf("Expected the key 'a' to be found")
	}
	if gt == photon {
		t.Errorf("Mutable cloning found")
	}
}

func TestSliceSubs(t *testing.T) {
	var sub Subs

	sub = newSliceSubs()
	if sub.Size() != 0 {
		t.Error("Expected a size of 0")
	}

	sub = newSliceSubs(5)
	if cap(sub.(*sSubs).s) != 5 {
		t.Error("Expected a cap of 5")
	}
	if sub.Size() != 0 {
		t.Error("Expected a size of 0")
	}

	testSubs(t, sub)

	// Format for completeness sake
	sub = newSliceSubs(2)
	sub = sub.Add('a', proton)
	sub = sub.Add('b', neutron)
	if fmt.Sprintf("%v", sub) != "{a: proton, b: neutron}" {
		t.Errorf("Format of sub is wrong. Got %q instead", sub)
	}
}

func TestMapSubs(t *testing.T) {
	var sub Subs

	sub = make(mSubs)
	if sub.Size() != 0 {
		t.Error("Expected a size of 0")
	}

	testSubs(t, sub)
}

var composeTests = []struct {
	a Subs
	b Subs

	expected Subs
}{
	{mSubs{'a': proton}, &sSubs{[]Substitution{{'b', neutron}}}, &sSubs{[]Substitution{{'a', proton}, {'b', neutron}}}},
	{&sSubs{[]Substitution{{'b', neutron}}}, mSubs{'a': proton}, mSubs{'a': proton, 'b': neutron}},

	{mSubs{'a': proton, 'b': neutron}, &sSubs{[]Substitution{{'b', neutron}}}, &sSubs{[]Substitution{{'a', proton}, {'b', neutron}}}},
	{mSubs{'a': proton, 'b': TypeVariable('a')}, &sSubs{[]Substitution{{'b', neutron}}}, &sSubs{[]Substitution{{'a', proton}, {'b', proton}}}},
	{mSubs{'a': proton}, &sSubs{[]Substitution{{'b', TypeVariable('a')}}}, &sSubs{[]Substitution{{'a', proton}, {'b', proton}}}},
}

func TestCompose(t *testing.T) {
	for i, cts := range composeTests {
		subs := compose(cts.a, cts.b)

		for _, v := range cts.expected.Iter() {
			if T, ok := subs.Get(v.Tv); !ok {
				t.Errorf("Test %d: Expected TypeVariable %v to be in subs", i, v.Tv)
			} else if T != v.T {
				t.Errorf("Test %d: Expected replacement to be %v. Got %v instead", i, v.T, T)
			}
		}
	}
}
